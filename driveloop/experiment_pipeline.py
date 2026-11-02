from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from driveloop.backends import DriveDreamer2Backend, MockGenerationBackend
from driveloop.runner import DriveLoopRunner
from driveloop.schema import DriveLoopConfig, DriveLoopRequest


@dataclass(frozen=True)
class ExperimentCase:
    name: str
    prompt: str
    condition: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    expected_condition: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentCase":
        if not data.get("name"):
            raise ValueError("experiment case requires a name")
        if not data.get("prompt"):
            raise ValueError(f"experiment case {data.get('name')} requires a prompt")
        return cls(
            name=str(data["name"]),
            prompt=str(data["prompt"]),
            metadata=dict(data.get("metadata", {})),
            condition=dict(data.get("condition", {})),
            tags=list(data.get("tags", [])),
            expected_condition=dict(data.get("expected_condition", {})),
        )


@dataclass(frozen=True)
class ExperimentPipelineConfig:
    max_iterations: int = 3
    target_score: float = 0.8
    backend_name: str = "mock"
    dd2_project_root: Any = "."
    dd2_config_name: str = "drivedreamer2_img_cond_mini_local"
    dd2_baseline_output_dir: Any = "/data/projects/DriveLoop/outputs/drivedreamer2_img_cond_mini"
    dd2_baseline_dataset_dir: Any = "/data/projects/DriveLoop/data/processed/nuscenes/v1.0-mini/cam_all_val/v0.0.2"
    dd2_audit_only: bool = False
    dd2_batch_skip: int = 0
    dd2_source_candidate_id: str | None = None
    dd2_sample_token: str | None = None
    dd2_scene_token: str | None = None
    dd2_instance_token: str | None = None
    dd2_source_identity_summary: Any = None
    dd2_timeout_seconds: int | None = None
    dd2_force_boxes3d_probe: bool = False
    dd2_boxes3d_probe_category: str | None = None
    dd2_frame_num: int = 8
    use_task_utility: bool = False
    utility_weights: Any = None
    perception_weights: Any = None
    perception_confidence: float = 0.25
    refiner_escalation: bool = True
    perception_baseline_video: Any = None
    # Perception protocol of record. "v9" is the single-class composite
    # evaluator; "v10a" adds super-class evidence and class fidelity; "v10b"
    # adds the maneuver view restriction (block 220-223). Default is "v10b":
    # the three-window rescore showed the v9 FT lever is a single-class
    # detection artifact that does not survive super-class pooling, so v10b is
    # the protocol of record. Acceptance is unaffected -- target_score is a
    # per-experiment knob (not a global calibrated tau), and the default 0.8 is
    # already unreachable under both v9 and v10, so both run to max_iterations
    # and record every attempt. Runs that need to reproduce the archived v9
    # numbers must set perception_protocol="v9" explicitly.
    perception_protocol: str = "v10b"
    acceptance_policy: str = "score_and_diagnosis"
    perception_acceptance_threshold: float = 0.3
    stop_on_acceptance: bool = True


def perception_evaluator_class(protocol: Any):
    """Resolve the perception evaluator class for a protocol name. Kept
    importable and detector-free so the selection can be unit-tested without
    loading YOLO weights."""
    name = str(protocol or "v9").lower()
    if name == "v9":
        from driveloop.composite_perception import CompositePerceptionVideoEvaluator
        return CompositePerceptionVideoEvaluator
    if name == "v10a":
        from driveloop.perception_v10 import SuperclassCompositePerceptionEvaluator
        return SuperclassCompositePerceptionEvaluator
    if name == "v10b":
        from driveloop.perception_v10 import ManeuverViewRestrictedSuperclassEvaluator
        return ManeuverViewRestrictedSuperclassEvaluator
    raise ValueError("unknown perception_protocol: %r" % (protocol,))


def load_experiment_cases(path: Path | str) -> list[ExperimentCase]:
    data = json.loads(Path(path).read_text())
    rows = data["cases"] if isinstance(data, dict) and "cases" in data else data
    if not isinstance(rows, list):
        raise ValueError("experiment manifest must be a list or contain a cases list")
    return [ExperimentCase.from_dict(row) for row in rows]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=_json_default) + "\n")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-").lower()
    return slug or "case"


class ExperimentPipeline:
    def __init__(
        self,
        output_dir: Path | str,
        config: ExperimentPipelineConfig | None = None,
        backend_factory: Callable[[Path], Any] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.config = config or ExperimentPipelineConfig()
        self.backend_factory = (
            backend_factory
            if backend_factory is not None
            else self._build_backend_factory(self.config)
        )

    def _build_backend_factory(self, config: ExperimentPipelineConfig) -> Callable[[Path], Any]:
        if config.backend_name == "mock":
            return lambda artifact_dir: MockGenerationBackend(output_dir=artifact_dir)
        if config.backend_name == "drivedreamer2":
            return lambda artifact_dir: DriveDreamer2Backend(
                project_root=config.dd2_project_root,
                config_name=config.dd2_config_name,
                artifact_dir=artifact_dir,
                baseline_output_dir=config.dd2_baseline_output_dir,
                baseline_dataset_dir=config.dd2_baseline_dataset_dir,
                audit_only=config.dd2_audit_only,
                batch_skip=config.dd2_batch_skip,
                source_candidate_id=config.dd2_source_candidate_id,
                sample_token=config.dd2_sample_token,
                scene_token=config.dd2_scene_token,
                instance_token=config.dd2_instance_token,
                source_identity_summary_path=config.dd2_source_identity_summary,
                timeout_seconds=config.dd2_timeout_seconds,
                force_boxes3d_probe=config.dd2_force_boxes3d_probe,
                boxes3d_probe_category=config.dd2_boxes3d_probe_category,
                source_selector_frame_num=config.dd2_frame_num,
            )
        raise ValueError(f"unsupported experiment backend: {config.backend_name}")

    def run_cases(self, cases: Iterable[ExperimentCase]) -> dict[str, Any]:
        case_list = list(cases)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        records = [self._run_case(case) for case in case_list]
        summary = {
            "schema_version": "driveloop_experiment_summary.v0",
            "backend": self.config.backend_name,
            "case_count": len(records),
            "accepted_count": sum(1 for record in records if record["status"] == "accepted"),
            "semantic_success_claim_allowed_count": sum(
                1 for record in records if record["claim_boundary"]["semantic_success_claim_allowed"]
            ),
            "claim_boundary": {
                "experiment_summary_is_not_video_semantic_success": True,
                "mock_backend_is_not_dd2_gpu_evidence": self.config.backend_name == "mock",
                "dd2_audit_only_is_not_video_semantic_success": (
                    self.config.backend_name == "drivedreamer2" and self.config.dd2_audit_only
                ),
                "semantic_success_requires_measured_passed_alignment_eval": True,
            },
            "cases": records,
        }
        _write_json(self.output_dir / "summary.json", summary)
        (self.output_dir / "summary.md").write_text(self._render_markdown(summary))
        return summary

    def _run_case(self, case: ExperimentCase) -> dict[str, Any]:
        case_dir = self.output_dir / _slugify(case.name)
        metadata = dict(case.metadata)
        metadata.setdefault("experiment_case", case.name)
        if case.tags:
            metadata.setdefault("experiment_tags", case.tags)
        if case.expected_condition:
            metadata.setdefault("expected_condition", case.expected_condition)

        evaluator = None
        if self.config.perception_weights:
            from driveloop.perception_video import UltralyticsYOLODetector
            evaluator_cls = perception_evaluator_class(self.config.perception_protocol)
            evaluator = evaluator_cls(
                detector=UltralyticsYOLODetector(
                    self.config.perception_weights,
                    confidence_threshold=self.config.perception_confidence,
                ),
                confidence_threshold=self.config.perception_confidence,
                baseline_video=self.config.perception_baseline_video,
            )
        refiner = None
        if not self.config.refiner_escalation:
            from driveloop.refiner import RuleBasedRefiner
            refiner = RuleBasedRefiner()
            refiner.PERCEPTION_ESCALATION = []  # saturated-refiner ablation
            refiner.STRUCTURAL_ESCALATION_ENABLED = False
        runner = DriveLoopRunner(
            backend=self.backend_factory(case_dir / "artifacts"),
            evaluator=evaluator,
            refiner=refiner,
            config=DriveLoopConfig(
                max_iterations=self.config.max_iterations,
                target_score=self.config.target_score,
                output_dir=case_dir / "history",
                use_task_utility=self.config.use_task_utility,
                utility_weights=self.config.utility_weights,
                acceptance_policy=self.config.acceptance_policy,
                perception_acceptance_threshold=self.config.perception_acceptance_threshold,
                stop_on_acceptance=self.config.stop_on_acceptance,
            ),
        )
        result = runner.run(
            DriveLoopRequest(prompt=case.prompt, condition=dict(case.condition), metadata=metadata)
        )
        attempt_records = [asdict(attempt) for attempt in result.attempt_history]

        _write_json(case_dir / "result.json", asdict(result))
        with (case_dir / "attempts.jsonl").open("w", encoding="utf-8") as f:
            for attempt in attempt_records:
                f.write(json.dumps(attempt, ensure_ascii=False, default=_json_default) + "\n")

        accepted = result.return_status == "accepted"
        returned_generation = result.returned_generation
        returned_evaluation = result.returned_evaluation
        backend_metadata = returned_generation.metadata
        dd2_audit_only = backend_metadata.get("dd2_audit_only") is True
        semantic_claim_allowed = (
            self.config.backend_name != "mock"
            and not dd2_audit_only
            and accepted
            and bool(returned_evaluation.diagnosis.passed)
        )
        record = {
            "schema_version": "driveloop_experiment_case_result.v0",
            "name": case.name,
            "tags": case.tags,
            "status": "accepted" if accepted else "failed",
            "best_score": result.best_evaluation.score,
            "best_metrics": dict(result.best_evaluation.metrics),
            "returned_score": returned_evaluation.score,
            "returned_metrics": dict(returned_evaluation.metrics),
            "returned_iteration": returned_generation.iteration,
            "return_status": result.return_status,
            "selection_metric": result.selection_metric,
            "acceptance_policy": result.acceptance_policy,
            "attempt_count": len(attempt_records),
            "attempt_statuses": [attempt.get("status") for attempt in attempt_records],
            "diagnosis_reasons": list(returned_evaluation.diagnosis.reasons),
            "suggested_actions": list(returned_evaluation.diagnosis.suggested_actions),
            "claim_boundary": {
                "experiment_case_record_is_not_video_semantic_success": True,
                "mock_backend_is_not_dd2_gpu_evidence": self.config.backend_name == "mock",
                "dd2_audit_only_is_not_video_semantic_success": dd2_audit_only,
                "semantic_success_requires_measured_passed_alignment_eval": True,
                "semantic_success_claim_allowed": semantic_claim_allowed,
                "source_selection_ready": any(
                    attempt.get("source_selection", {}).get("ready") is True
                    for attempt in attempt_records
                ),
                "dd2_source_sample_binding_ready": (
                    backend_metadata.get("dd2_source_sample_binding", {}).get("ready") is True
                    if isinstance(backend_metadata.get("dd2_source_sample_binding"), dict)
                    else False
                ),
                "perception_evaluation_enabled": bool(
                    case.metadata.get("perception_evaluation", {}).get("enabled")
                    if isinstance(case.metadata.get("perception_evaluation"), dict)
                    else False
                ),
            },
            "outputs": {
                "case_dir": str(case_dir),
                "result_json": str(case_dir / "result.json"),
                "attempts_jsonl": str(case_dir / "attempts.jsonl"),
                "history_jsonl": str(case_dir / "history" / "history.jsonl"),
            },
        }
        _write_json(case_dir / "case_summary.json", record)
        return record

    def _render_markdown(self, summary: dict[str, Any]) -> str:
        lines = [
            "# DriveLoop Experiment Summary",
            "",
            f"- schema_version: `{summary['schema_version']}`",
            f"- backend: `{summary['backend']}`",
            f"- case_count: `{summary['case_count']}`",
            f"- accepted_count: `{summary['accepted_count']}`",
            f"- semantic_success_claim_allowed_count: `{summary['semantic_success_claim_allowed_count']}`",
            "",
            "| case | status | best_score | attempts | semantic_success_claim_allowed |",
            "| --- | --- | ---: | ---: | --- |",
        ]
        for row in summary["cases"]:
            lines.append(
                f"| {row['name']} | {row['status']} | {row['best_score']:.3f} | "
                f"{row['attempt_count']} | {row['claim_boundary']['semantic_success_claim_allowed']} |"
            )
        lines.extend([
            "",
            "Claim boundary: this file summarizes experiment records only; it is not video semantic-success evidence.",
            "",
        ])
        return "\n".join(lines)
