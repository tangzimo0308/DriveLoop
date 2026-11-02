from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Optional

from driveloop.backends.base import GenerationBackend
from driveloop.condition_adapter import DriveDreamer2ConditionAdapter
from driveloop.evaluator import BaseEvaluator, RuleBasedEvaluator
from driveloop.grounding import RuleBasedGrounder
from driveloop.intent.providers import MultimodalPreprocessor
from driveloop.logging import HistoryLogger
from driveloop.longtail import LongTailController, control_coverage
from driveloop.refiner import RuleBasedRefiner
from driveloop.schema import (
    DriveLoopAttempt,
    Diagnosis,
    DriveLoopConfig,
    DriveLoopRequest,
    DriveLoopResult,
    Evaluation,
    Generation,
    Refinement,
)
from driveloop.source_selector import BaseSourceSelector, NoOpSourceSelector
from driveloop.control_visibility import control_visibility_score
from driveloop.utility import UtilityWeights, intent_consistency, task_utility
from driveloop.acceptance import POLICIES, acceptance_state, finite_number


class DriveLoopRunner:
    def __init__(
        self,
        backend: GenerationBackend,
        evaluator: Optional[BaseEvaluator] = None,
        refiner: Optional[RuleBasedRefiner] = None,
        grounder: Optional[RuleBasedGrounder] = None,
        longtail_controller: Optional[LongTailController] = None,
        condition_adapter: Optional[DriveDreamer2ConditionAdapter] = None,
        source_selector: Optional[BaseSourceSelector] = None,
        config: Optional[DriveLoopConfig] = None,
    ) -> None:
        self.backend = backend
        self.evaluator = evaluator or RuleBasedEvaluator()
        self.refiner = refiner or RuleBasedRefiner()
        self.grounder = grounder or RuleBasedGrounder(
            multimodal_preprocessor=MultimodalPreprocessor()
        )
        self.longtail_controller = longtail_controller or LongTailController()
        self.condition_adapter = condition_adapter or DriveDreamer2ConditionAdapter()
        self.source_selector = source_selector or NoOpSourceSelector()
        self.config = config or DriveLoopConfig()
        if self.config.acceptance_policy not in POLICIES:
            raise ValueError("unknown acceptance_policy: %r" % self.config.acceptance_policy)
        if self.config.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        for threshold in (self.config.target_score, self.config.perception_acceptance_threshold):
            if finite_number(threshold) is None or not 0 <= threshold <= 1:
                raise ValueError("acceptance thresholds must be finite and in [0, 1]")
        self.history_logger = HistoryLogger(self.config.output_dir)

    def run(self, request: DriveLoopRequest) -> DriveLoopResult:
        current_request = request
        original_spec = self.grounder.ground(request)
        history: list[tuple[Generation, Evaluation]] = []
        attempt_history: list[DriveLoopAttempt] = []
        best_generation: Optional[Generation] = None
        best_evaluation: Optional[Evaluation] = None
        best_accepted = None

        for iteration in range(self.config.max_iterations):
            scene_spec = self.grounder.ground(current_request)
            requested_tags = current_request.condition.get("long_tail_tags", [])
            condition_plan = self.longtail_controller.build(
                scene_spec,
                requested_tags=requested_tags,
                history=history,
            )
            structural_escalation = current_request.condition.get("structural_escalation")
            if isinstance(structural_escalation, dict) and structural_escalation:
                condition_plan = replace(
                    condition_plan,
                    executable_controls={
                        **condition_plan.executable_controls,
                        "structural_escalation": structural_escalation,
                    },
                )
            coverage = control_coverage(condition_plan)
            dd2_condition = self.condition_adapter.build(
                scene_spec,
                condition_plan,
                alignment_feedback=current_request.condition.get("alignment_feedback"),
            )
            source_selection = self.source_selector.select(current_request, scene_spec, condition_plan)
            source_selection_dict = asdict(source_selection)
            dd2_condition_dict = asdict(dd2_condition)
            generation_request = self._with_conditioned_prompt(current_request, condition_plan.prompt_suffixes)
            generation_request = replace(
                generation_request,
                condition={
                    **generation_request.condition,
                    "dd2_condition": dd2_condition_dict,
                },
                metadata={
                    **generation_request.metadata,
                    "source_selection": source_selection_dict,
                    **source_selection.backend_hints,
                },
            )
            generation = self.backend.generate(generation_request, iteration)
            if self.config.use_postprocess and condition_plan.postprocess_effects:
                from driveloop.postprocess import apply_postprocess_effects
                raw_video = generation.artifacts.get("video")
                if raw_video:
                    post_report = apply_postprocess_effects(raw_video, condition_plan.postprocess_effects)
                    generation = replace(
                        generation,
                        artifacts={
                            **generation.artifacts,
                            "video_pre_postprocess": raw_video,
                            "video": post_report["output"],
                        },
                        metadata={**generation.metadata, "postprocess_report": post_report},
                    )
                else:
                    generation = replace(
                        generation,
                        metadata={
                            **generation.metadata,
                            "postprocess_report": {
                                "applied": [],
                                "skipped_unknown": list(condition_plan.postprocess_effects),
                                "reason": "no_video_artifact",
                            },
                        },
                    )
            perception_metadata = self._perception_metadata_from_request(current_request.metadata)
            generation = replace(
                generation,
                metadata={
                    **generation.metadata,
                    **perception_metadata,
                    "source_selection": source_selection_dict,
                    "scene_specification": asdict(scene_spec),
                    "long_tail_condition_plan": asdict(condition_plan),
                    "control_coverage": coverage,
                    "dd2_condition": dd2_condition_dict,
                },
            )

            reset_audit = getattr(getattr(self.evaluator, "detector", None), "begin_runtime_audit", None)
            if callable(reset_audit):
                reset_audit()
            evaluation = self.evaluator.evaluate(generation)
            generation = replace(
                generation,
                metadata={
                    **generation.metadata,
                    **self._perception_support_provenance(generation),
                    **self._perception_runtime_provenance(),
                },
            )
            release_gpu = getattr(getattr(self.evaluator, "detector", None), "release_gpu", None)
            if callable(release_gpu):
                release_gpu()
            evaluation = self._with_source_selection_diagnosis(evaluation, source_selection_dict)
            if self.config.use_task_utility:
                utility_weights = (
                    UtilityWeights(**self.config.utility_weights)
                    if self.config.utility_weights
                    else None
                )
                alignment_score = evaluation.metrics.get("alignment_score")
                alignment_source = "measured_alignment"
                auto_ctrl = None
                if alignment_score is None:
                    auto_ctrl = control_visibility_score(
                        evaluation.metrics, scene_spec, condition_plan
                    )
                    if auto_ctrl["score"] is not None:
                        alignment_score = auto_ctrl["score"]
                        alignment_source = auto_ctrl["source"]
                utility = task_utility(
                    evaluation.score,
                    condition_plan,
                    original_spec,
                    scene_spec,
                    alignment_score=alignment_score,
                    weights=utility_weights,
                    alignment_source=alignment_source,
                )
                generation = replace(
                    generation,
                    metadata={
                        **generation.metadata,
                        "task_utility": utility,
                        "control_visibility": auto_ctrl,
                    },
                )
                evaluation = replace(
                    evaluation,
                    score=utility["J"],
                    metrics={
                        **evaluation.metrics,
                        "J": utility["J"],
                        "S_perc": utility["S_perc"],
                        "S_ctrl": utility["S_ctrl"],
                        "S_intent": utility["S_intent"],
                    },
                )
            history.append((generation, evaluation))

            if best_evaluation is None or evaluation.score > best_evaluation.score:
                best_generation = generation
                best_evaluation = evaluation

            status = self._attempt_status(generation, evaluation, source_selection_dict)
            if status == "accepted" and (
                best_accepted is None or evaluation.score > best_accepted[1].score
            ):
                best_accepted = (generation, evaluation)
            should_stop = (
                status == "accepted" and self.config.stop_on_acceptance
            ) or self._perception_unmeasurable(evaluation)
            refinement: Optional[Refinement] = None
            if not should_stop:
                refinement = self.refiner.refine(current_request, evaluation)
                if self.config.use_task_utility and refinement is not None:
                    refined_spec = self.grounder.ground(
                        replace(current_request, prompt=refinement.prompt)
                    )
                    drift_next = 1.0 - intent_consistency(original_spec, refined_spec)
                    drift_now = 1.0 - intent_consistency(original_spec, scene_spec)
                    if drift_next > drift_now + self.config.intent_epsilon:
                        refinement = Refinement(
                            prompt=current_request.prompt,
                            condition=dict(refinement.condition),
                            notes=list(refinement.notes)
                            + ["intent_guard_reverted_prompt (Eq.18)"],
                        )

            attempt = DriveLoopAttempt(
                iteration=iteration,
                request=current_request,
                scene_specification=scene_spec,
                long_tail_condition_plan=condition_plan,
                dd2_condition=dd2_condition_dict,
                condition_package=self._attempt_condition_package(dd2_condition_dict, generation),
                source_binding=self._source_binding_from_generation(generation),
                generation=generation,
                evaluation=evaluation,
                refinement=refinement,
                status=status,
                source_selection=source_selection_dict,
                claim_boundary=self._attempt_claim_boundary(generation, evaluation, source_selection_dict),
            )
            attempt_history.append(attempt)
            self.history_logger.write(generation, evaluation, attempt=attempt)

            if should_stop:
                break

            assert refinement is not None
            current_request = replace(
                current_request,
                prompt=refinement.prompt,
                condition=refinement.condition,
            )

        assert best_generation is not None
        assert best_evaluation is not None
        returned_generation, returned_evaluation = best_accepted or (best_generation, best_evaluation)
        return_status = "accepted" if best_accepted else "budget_exhausted"
        if best_accepted is None:
            returned_attempt = next(a for a in attempt_history if a.iteration == returned_generation.iteration)
            if returned_attempt.status in ("requires_review", "perception_unmeasurable",
                                            "source_selection_unavailable", "source_binding_unavailable"):
                return_status = returned_attempt.status
        return DriveLoopResult(
            request=request,
            best_generation=best_generation,
            best_evaluation=best_evaluation,
            history=history,
            attempt_history=attempt_history,
            returned_generation=returned_generation,
            returned_evaluation=returned_evaluation,
            return_status=return_status,
            selection_metric="J" if self.config.use_task_utility else "evaluation_score",
            acceptance_policy=self.config.acceptance_policy,
        )

    def _with_source_selection_diagnosis(self, evaluation: Evaluation, source_selection: dict) -> Evaluation:
        if source_selection.get("requested") is not True or source_selection.get("ready") is True:
            return evaluation

        diagnosis = source_selection.get("diagnosis", {})
        if not isinstance(diagnosis, dict):
            diagnosis = {}
        binding = source_selection.get("binding", {})
        if not isinstance(binding, dict):
            binding = {}

        reasons = list(evaluation.diagnosis.reasons)
        reasons.append("source_selection_unavailable")
        for reason in (diagnosis.get("reason"), binding.get("reason")):
            if reason:
                reasons.append(str(reason))

        actions = list(evaluation.diagnosis.suggested_actions)
        suggested_actions = diagnosis.get("suggested_actions", [])
        if isinstance(suggested_actions, list):
            actions.extend(str(action) for action in suggested_actions)

        metrics = {
            **evaluation.metrics,
            "source_selection_requested": 1.0,
            "source_selection_ready": 0.0,
        }

        return replace(
            evaluation,
            score=0.0,
            metrics=metrics,
            diagnosis=Diagnosis(
                passed=False,
                reasons=list(dict.fromkeys(reasons)),
                suggested_actions=list(dict.fromkeys(actions)),
            ),
        )

    def _perception_runtime_provenance(self) -> dict:
        audit = getattr(getattr(self.evaluator, "detector", None), "get_runtime_audit", None)
        if not callable(audit):
            return {}
        return {"perception_detector_runtime": {
            **audit(),
            "evaluator_class": type(self.evaluator).__name__,
            "evaluator_post_filter_confidence_threshold": getattr(self.evaluator, "confidence_threshold", None),
            "tracker_iou_threshold": getattr(self.evaluator, "tracker_iou_threshold", None),
        }}

    def _perception_support_provenance(self, generation: Generation) -> dict:
        """Archive which no-injection support video the evaluator resolved for
        this generation. Without it the support term is only recoverable by
        fingerprinting subtracted-detection counters (2026-07-18 audit)."""
        resolver = getattr(self.evaluator, "resolve_baseline_video", None)
        if not callable(resolver):
            return {}
        resolved = resolver(generation)
        if resolved is None:
            return {
                "perception_baseline_video_resolved": None,
                "perception_baseline_video_exists": False,
            }
        return {
            "perception_baseline_video_resolved": resolved,
            "perception_baseline_video_exists": Path(resolved).exists(),
        }

    def _perception_metadata_from_request(self, metadata: dict) -> dict:
        config = metadata.get("perception_evaluation")
        if not isinstance(config, dict) or config.get("enabled") is not True:
            return {}

        payload = {"perception_evaluation": dict(config)}
        for source in (metadata, config):
            if not isinstance(source, dict):
                continue
            for key in (
                "perception_detections",
                "detections_by_frame",
                "video_detections",
                "target_labels",
                "frame_count",
            ):
                if key in source:
                    payload[key] = source[key]

        if "detections" in config and "perception_detections" not in payload:
            payload["perception_detections"] = config["detections"]
        return payload

    def _source_binding_from_generation(self, generation: Generation) -> dict:
        binding = generation.metadata.get("dd2_source_sample_binding", {})
        return dict(binding) if isinstance(binding, dict) else {}

    def _attempt_condition_package(self, dd2_condition: dict, generation: Generation) -> dict:
        executable = dd2_condition.get("executable_condition", {})
        if not isinstance(executable, dict):
            executable = {}
        structural_plan = executable.get("structural_input_plan", {})
        if not isinstance(structural_plan, dict):
            structural_plan = {}
        source_binding = self._source_binding_from_generation(generation)
        baseline_snapshot = generation.metadata.get("dd2_baseline_structural_snapshot", {})
        if not isinstance(baseline_snapshot, dict):
            baseline_snapshot = {}
        long_tail_tags = dd2_condition.get("long_tail_tags", [])
        motion_primitives = dd2_condition.get("motion_primitives", [])
        return {
            "schema_version": "driveloop_attempt_condition_package.v0",
            "target_backend": executable.get("target_backend"),
            "text_prompt": dd2_condition.get("text_prompt"),
            "long_tail_tags": list(long_tail_tags) if isinstance(long_tail_tags, list) else [],
            "motion_primitives": list(motion_primitives) if isinstance(motion_primitives, list) else [],
            "source_binding_requested": source_binding.get("requested"),
            "source_binding_ready": source_binding.get("ready"),
            "dd2_batch_skip": source_binding.get("dd2_batch_skip", generation.metadata.get("dd2_batch_skip")),
            "runtime_dataset_dir": source_binding.get("dataset_dir") or baseline_snapshot.get("dataset_dir"),
            "unsupported_controls": self._unsupported_controls(executable, structural_plan),
        }

    def _unsupported_controls(self, executable: dict, structural_plan: dict) -> list[str]:
        unsupported: list[str] = []
        trajectory = executable.get("trajectory_control_contract", {})
        if isinstance(trajectory, dict) and trajectory.get("status") == "not_runtime_connected":
            unsupported.append("trajectory_control")
        for name in ("boxes3d", "image_box", "image_hdmap"):
            control = structural_plan.get(name, {})
            if isinstance(control, dict) and control.get("override_ready") is False:
                unsupported.append(name)
        return list(dict.fromkeys(unsupported))

    @staticmethod
    def _perception_unmeasurable(evaluation: Evaluation) -> bool:
        """True when the perception protocol could not measure the case at all
        (v10b's maneuver view restriction resolved to no scorable view because
        the case carries no actor_motion_surface_plan). This is distinct from a
        low score: the case is unmeasurable, not failed, so the loop must not
        churn refinements against it and it must not be recorded as a semantic
        failure. Inert under v9, which sets no such flag."""
        return bool(evaluation.metrics.get("perception_view_restriction_unresolved"))

    def _attempt_status(self, generation: Generation, evaluation: Evaluation, source_selection: dict) -> str:
        if source_selection.get("requested") is True and source_selection.get("ready") is not True:
            return "source_selection_unavailable"
        source_binding = self._source_binding_from_generation(generation)
        if source_binding.get("requested") is True and source_binding.get("ready") is not True:
            return "source_binding_unavailable"
        if self._perception_unmeasurable(evaluation):
            return "perception_unmeasurable"
        gate = acceptance_state(evaluation, self.config)
        if gate == "PASS":
            return "accepted"
        if gate == "UNMEASURED":
            return "requires_review"
        return "needs_refinement"

    def _attempt_claim_boundary(self, generation: Generation, evaluation: Evaluation, source_selection: dict) -> dict:
        source_binding = self._source_binding_from_generation(generation)
        source_claim = source_binding.get("claim_boundary", {}) if isinstance(source_binding, dict) else {}
        selection_claim = source_selection.get("claim_boundary", {}) if isinstance(source_selection, dict) else {}
        return {
            "attempt_record_is_not_video_semantic_success": True,
            "generation_artifact_is_not_semantic_success": True,
            "runtime_trace_is_not_semantic_success": True,
            "semantic_success_requires_measured_passed_alignment_eval": True,
            "source_binding_is_not_gpu_approval": source_claim.get("source_sample_binding_is_not_gpu_approval"),
            "source_selection_is_not_gpu_approval": selection_claim.get("source_selection_is_not_gpu_approval"),
            "evaluation_passed": evaluation.diagnosis.passed,
            "acceptance_policy": self.config.acceptance_policy,
            "acceptance_state": acceptance_state(evaluation, self.config),
            "selection_metric": "J" if self.config.use_task_utility else "evaluation_score",
            "perception_acceptance_threshold": self.config.perception_acceptance_threshold,
            "stop_on_acceptance": self.config.stop_on_acceptance,
        }

    def _with_conditioned_prompt(
        self,
        request: DriveLoopRequest,
        prompt_suffixes: list[str],
    ) -> DriveLoopRequest:
        if not prompt_suffixes:
            return request
        suffix = ", ".join(prompt_suffixes)
        prompt = request.prompt.rstrip(".")
        if suffix.lower() in prompt.lower():
            return request
        return replace(request, prompt=f"{prompt}, {suffix}.")
