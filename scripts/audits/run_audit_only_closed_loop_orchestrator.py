from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from scripts.run_alignment_failure_taxonomy import build_taxonomy
    from scripts.run_closed_loop_case_summary import alignment_summary, as_dict, as_list, format_value, load_json
except ModuleNotFoundError:
    from run_alignment_failure_taxonomy import build_taxonomy
    from run_closed_loop_case_summary import alignment_summary, as_dict, as_list, format_value, load_json


def is_accepted(summary: dict[str, Any], target_score: float) -> bool:
    score = summary.get("score")
    score_passed = isinstance(score, (float, int)) and float(score) >= target_score
    diagnosis_passed = summary.get("passed") is True
    claim_passed = summary.get("video_semantic_claim") == "measured_passed"
    return claim_passed or diagnosis_passed or score_passed


def generation_prompt(alignment_eval: dict[str, Any]) -> str | None:
    generation = as_dict(alignment_eval.get("generation"))
    prompt = generation.get("prompt")
    return str(prompt) if prompt else None


def derive_refinement_prompt(prompt: str | None, taxonomy: dict[str, Any]) -> str | None:
    if not prompt:
        return None

    labels = set(as_list(taxonomy.get("taxonomy_labels")))
    suffixes: list[str] = []

    if labels & {"object_identity_failed", "motorcycle_identity_failed", "low_visual_confidence"}:
        suffixes.append("the target actor remains large, visible, high-contrast, and unoccluded")
    if labels & {"tracking_identity_failed"}:
        suffixes.append("the same target actor remains trackable across all reviewed frames")
    if labels & {"cut_in_motion_failed", "lane_change_motion_failed", "lateral_motion_failed", "spatial_relation_failed"}:
        suffixes.append("the target actor shows measurable lateral motion from an adjacent lane toward the ego path")
    if labels & {"hdmap_alignment_failed", "road_marking_conflict"}:
        suffixes.append("visible lane geometry remains consistent with the requested maneuver")

    if not suffixes:
        suffixes.append("make the requested objects, relations, and motion visually measurable")

    existing = prompt.rstrip(". ")
    return existing + ", " + ", ".join(suffixes) + "."


def build_refinement_proposal(
    failed_alignment_eval: dict[str, Any],
    taxonomy: dict[str, Any],
    provided_proposal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if provided_proposal:
        return {
            "source": "provided_refinement_proposal",
            "available": True,
            "source_prompt": provided_proposal.get("source_prompt"),
            "refined_prompt": provided_proposal.get("refined_prompt"),
            "retry_policy": as_dict(provided_proposal.get("retry_policy")),
            "does_not_run_gpu": provided_proposal.get("does_not_run_gpu"),
            "semantic_success_claim_allowed": provided_proposal.get("semantic_success_claim_allowed"),
            "claim_boundary": {
                "proposal_is_audit_only": True,
                "proposal_is_not_gpu_approval": True,
                "proposal_is_not_video_semantic_success": True,
            },
        }

    prompt = generation_prompt(failed_alignment_eval)
    return {
        "source": "derived_from_failure_taxonomy",
        "available": True,
        "source_prompt": prompt,
        "refined_prompt": derive_refinement_prompt(prompt, taxonomy),
        "taxonomy_labels": as_list(taxonomy.get("taxonomy_labels")),
        "retry_policy": {
            "explicit_gpu_retry_approval_required": True,
            "post_gpu_review_required_after_any_retry": True,
            "proposal_is_not_gpu_approval": True,
        },
        "does_not_run_gpu": True,
        "semantic_success_claim_allowed": False,
        "claim_boundary": {
            "proposal_is_audit_only": True,
            "proposal_is_not_gpu_approval": True,
            "proposal_is_not_video_semantic_success": True,
        },
    }


def attempt_record(
    iteration: int,
    alignment_eval: dict[str, Any],
    target_score: float,
    role: str,
) -> dict[str, Any]:
    summary = alignment_summary(alignment_eval)
    accepted = is_accepted(summary, target_score)
    return {
        "iteration": iteration,
        "role": role,
        "status": "accepted" if accepted else "needs_diagnosis_and_refinement",
        "alignment": summary,
        "accepted": accepted,
    }


def build_orchestrator_trace(
    initial_alignment_eval: dict[str, Any],
    retry_alignment_eval: dict[str, Any] | None = None,
    failure_taxonomy: dict[str, Any] | None = None,
    refinement_proposal: dict[str, Any] | None = None,
    candidate_audit: dict[str, Any] | None = None,
    case_id: str = "audit_only_closed_loop",
    target_score: float = 0.9,
    max_iterations: int = 2,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    algorithm_trace: list[dict[str, Any]] = []

    algorithm_trace.append({"step": "initialize_history", "status": "completed"})
    algorithm_trace.append({"step": "evaluate_attempt_0", "status": "completed"})

    initial_attempt = attempt_record(0, initial_alignment_eval, target_score, "pre_refinement")
    attempts.append(initial_attempt)

    taxonomy = failure_taxonomy
    proposal: dict[str, Any] | None = None

    if initial_attempt["accepted"]:
        closed_loop_status = "accepted_without_refinement"
        algorithm_trace.append({"step": "accept_attempt_0", "status": "completed"})
    else:
        taxonomy = taxonomy or build_taxonomy(initial_alignment_eval, candidate_audit or {})
        proposal = build_refinement_proposal(initial_alignment_eval, taxonomy, refinement_proposal)
        algorithm_trace.extend(
            [
                {"step": "diagnose_failure", "status": "completed"},
                {"step": "refine_prompt_or_condition", "status": "completed"},
                {
                    "step": "regenerate",
                    "status": "blocked_requires_explicit_generation_step",
                    "reason": "audit_only_orchestrator_does_not_run_gpu_or_dd2",
                },
            ]
        )

        if retry_alignment_eval and max_iterations > 1:
            algorithm_trace.append({"step": "evaluate_attempt_1", "status": "completed"})
            retry_attempt = attempt_record(1, retry_alignment_eval, target_score, "post_refinement_retry")
            attempts.append(retry_attempt)
            closed_loop_status = (
                "measured_failed_to_measured_passed"
                if retry_attempt["accepted"]
                else "refined_retry_not_accepted"
            )
        else:
            closed_loop_status = "diagnosed_and_refinement_proposed_waiting_for_generation"

    return {
        "schema_version": "driveloop_audit_only_closed_loop_orchestrator.v0",
        "case_id": case_id,
        "target_score": target_score,
        "max_iterations": max_iterations,
        "closed_loop_status": closed_loop_status,
        "attempts": attempts,
        "failure_taxonomy": taxonomy or {},
        "refinement_proposal": proposal or build_refinement_proposal(initial_alignment_eval, taxonomy or {}, refinement_proposal)
        if not initial_attempt["accepted"]
        else {},
        "algorithm_trace": algorithm_trace,
        "claim_boundary": {
            "orchestrator_does_not_generate_video": True,
            "orchestrator_does_not_run_gpu": True,
            "orchestrator_does_not_call_dd2": True,
            "video_or_tensor_existence_is_not_semantic_success": True,
            "semantic_success_requires_measured_alignment_review": True,
            "regeneration_requires_separate_explicit_approval": True,
        },
    }


def render_markdown(trace: dict[str, Any]) -> str:
    lines = [
        f"# Audit-only Closed-loop Trace: {trace.get('case_id')}",
        "",
        f"- Status: `{trace.get('closed_loop_status')}`",
        f"- Target score: `{trace.get('target_score')}`",
        "",
        "## Attempts",
        "",
        "| Iteration | Role | Claim | Score | Checks | Accepted |",
        "|---:|---|---:|---:|---:|---:|",
    ]

    for attempt in as_list(trace.get("attempts")):
        alignment = as_dict(attempt.get("alignment"))
        checks = f"{format_value(alignment.get('passed_required_check_count'))}/{format_value(alignment.get('required_check_count'))}"
        lines.append(
            f"| {attempt.get('iteration')} | {attempt.get('role')} | "
            f"`{alignment.get('video_semantic_claim')}` | {format_value(alignment.get('score'))} | "
            f"{checks} | {attempt.get('accepted')} |"
        )

    lines.extend(["", "## Algorithm Trace", ""])
    for step in as_list(trace.get("algorithm_trace")):
        reason = step.get("reason")
        suffix = f" - {reason}" if reason else ""
        lines.append(f"- `{step.get('step')}`: `{step.get('status')}`{suffix}")

    labels = as_list(as_dict(trace.get("failure_taxonomy")).get("taxonomy_labels"))
    lines.extend(["", "## Failure Labels", ""])
    lines.extend(f"- `{label}`" for label in labels) if labels else lines.append("- not_available")

    lines.extend(["", "## Claim Boundary", ""])
    for key, value in as_dict(trace.get("claim_boundary")).items():
        lines.append(f"- `{key}`: `{value}`")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an audit-only DriveLoop closed-loop orchestration trace.")
    parser.add_argument("--case-id", default="audit_only_closed_loop")
    parser.add_argument("--initial-alignment-eval", required=True, type=Path)
    parser.add_argument("--retry-alignment-eval", type=Path, default=None)
    parser.add_argument("--failure-taxonomy", type=Path, default=None)
    parser.add_argument("--refinement-proposal", type=Path, default=None)
    parser.add_argument("--candidate-audit", type=Path, default=None)
    parser.add_argument("--target-score", type=float, default=0.9)
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    trace = build_orchestrator_trace(
        initial_alignment_eval=load_json(args.initial_alignment_eval),
        retry_alignment_eval=load_json(args.retry_alignment_eval) if args.retry_alignment_eval else None,
        failure_taxonomy=load_json(args.failure_taxonomy) if args.failure_taxonomy else None,
        refinement_proposal=load_json(args.refinement_proposal) if args.refinement_proposal else None,
        candidate_audit=load_json(args.candidate_audit) if args.candidate_audit else None,
        case_id=args.case_id,
        target_score=args.target_score,
        max_iterations=args.max_iterations,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    print(args.output_json)

    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(trace), encoding="utf-8")
        print(args.output_md)

    print(json.dumps(trace, indent=2))


if __name__ == "__main__":
    main()
