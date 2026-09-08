from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any


DEFAULT_BASELINE_OUTPUT_DIR = "/data/projects/DriveLoop/outputs/drivedreamer2_img_cond_mini"


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def shell_command(args: list[str]) -> str:
    return " \\\n  ".join(shlex.quote(str(arg)) for arg in args)


def has_blocked_regeneration_step(trace: dict[str, Any]) -> bool:
    for step in as_list(trace.get("algorithm_trace")):
        step = as_dict(step)
        if step.get("step") == "regenerate" and "blocked" in str(step.get("status", "")):
            return True
    return False


def source_binding_summary(readiness: dict[str, Any] | None) -> dict[str, Any]:
    readiness = readiness or {}
    runtime = as_dict(readiness.get("runtime_binding_assessment"))
    binding = as_dict(runtime.get("source_sample_binding"))

    if not binding:
        binding = as_dict(readiness.get("source_sample_binding"))
    if not binding and ("dataset_dir" in readiness or "selector" in readiness):
        binding = readiness

    selector = as_dict(binding.get("selector"))
    return {
        "available": bool(binding),
        "ready": binding.get("ready"),
        "dataset_dir": binding.get("dataset_dir"),
        "source_candidate_id": first_present(selector.get("source_candidate_id"), binding.get("source_candidate_id")),
        "instance_token": first_present(selector.get("instance_token"), binding.get("instance_token")),
        "identity_summary_path": first_present(
            selector.get("identity_summary_path"),
            binding.get("identity_summary_path"),
            binding.get("source_identity_summary_path"),
        ),
    }


def proposal_from_trace(trace: dict[str, Any]) -> dict[str, Any]:
    proposal = as_dict(trace.get("refinement_proposal"))
    return {
        "available": bool(proposal),
        "source": proposal.get("source"),
        "source_prompt": proposal.get("source_prompt"),
        "refined_prompt": proposal.get("refined_prompt"),
        "retry_policy": as_dict(proposal.get("retry_policy")),
        "does_not_run_gpu": proposal.get("does_not_run_gpu"),
        "semantic_success_claim_allowed": proposal.get("semantic_success_claim_allowed"),
    }


def build_command_args(
    *,
    prompt: str | None,
    scenario_id: str,
    output_dir: str,
    config_name: str,
    dd2_batch_skip: int,
    target_score: float,
    max_iterations: int,
    source_candidate_id: str | None,
    instance_token: str | None,
    source_identity_summary: str | None,
    baseline_dataset_dir: str | None,
    baseline_output_dir: str,
) -> list[str]:
    if not prompt:
        return []

    command = [
        "python",
        "scripts/run_driveloop_drivedreamer2.py",
        "--prompt",
        prompt,
        "--scenario-id",
        scenario_id,
        "--max-iterations",
        str(max_iterations),
        "--target-score",
        str(target_score),
        "--output-dir",
        output_dir,
        "--config-name",
        config_name,
        "--dd2-batch-skip",
        str(dd2_batch_skip),
    ]

    optional_pairs = [
        ("--source-candidate-id", source_candidate_id),
        ("--instance-token", instance_token),
        ("--source-identity-summary", source_identity_summary),
        ("--baseline-dataset-dir", baseline_dataset_dir),
        ("--baseline-output-dir", baseline_output_dir),
    ]
    for flag, value in optional_pairs:
        if value:
            command.extend([flag, str(value)])

    return command


def build_regeneration_command_plan(
    orchestrator_trace: dict[str, Any],
    source_binding_readiness: dict[str, Any] | None = None,
    *,
    scenario_id: str | None = None,
    output_dir: str | None = None,
    config_name: str = "drivedreamer2_img_cond_mini_local",
    dd2_batch_skip: int = 0,
    target_score: float = 0.9,
    max_iterations: int = 1,
    source_candidate_id: str | None = None,
    instance_token: str | None = None,
    source_identity_summary: str | None = None,
    baseline_dataset_dir: str | None = None,
    baseline_output_dir: str = DEFAULT_BASELINE_OUTPUT_DIR,
) -> dict[str, Any]:
    case_id = str(orchestrator_trace.get("case_id") or "audit_only_closed_loop")
    scenario_id = scenario_id or f"{case_id}_gpu_retry_draft"
    output_dir = output_dir or f"outputs/driveloop/{scenario_id}"

    proposal = proposal_from_trace(orchestrator_trace)
    binding = source_binding_summary(source_binding_readiness)

    prompt = proposal.get("refined_prompt")
    resolved_source_candidate_id = first_present(source_candidate_id, binding.get("source_candidate_id"))
    resolved_instance_token = first_present(instance_token, binding.get("instance_token"))
    resolved_identity_summary = first_present(source_identity_summary, binding.get("identity_summary_path"))
    resolved_baseline_dataset = first_present(baseline_dataset_dir, binding.get("dataset_dir"))

    command_args = build_command_args(
        prompt=prompt,
        scenario_id=scenario_id,
        output_dir=output_dir,
        config_name=config_name,
        dd2_batch_skip=dd2_batch_skip,
        target_score=target_score,
        max_iterations=max_iterations,
        source_candidate_id=resolved_source_candidate_id,
        instance_token=resolved_instance_token,
        source_identity_summary=resolved_identity_summary,
        baseline_dataset_dir=resolved_baseline_dataset,
        baseline_output_dir=baseline_output_dir,
    )

    missing_required = []
    if not prompt:
        missing_required.append("refined_prompt")
    if not has_blocked_regeneration_step(orchestrator_trace):
        missing_required.append("blocked_regeneration_step")

    missing_recommended = []
    for name, value in [
        ("source_candidate_id", resolved_source_candidate_id),
        ("instance_token", resolved_instance_token),
        ("source_identity_summary", resolved_identity_summary),
        ("baseline_dataset_dir", resolved_baseline_dataset),
    ]:
        if not value:
            missing_recommended.append(name)

    if missing_required:
        command_status = "draft_incomplete_missing_required_fields"
    else:
        command_status = "draft_ready_for_explicit_approval"

    return {
        "schema_version": "driveloop_audit_only_regeneration_command_plan.v0",
        "case_id": case_id,
        "command_status": command_status,
        "missing_required_fields": missing_required,
        "missing_recommended_fields": missing_recommended,
        "orchestrator_status": orchestrator_trace.get("closed_loop_status"),
        "regeneration_blocked_in_trace": has_blocked_regeneration_step(orchestrator_trace),
        "resolved_inputs": {
            "prompt": prompt,
            "scenario_id": scenario_id,
            "output_dir": output_dir,
            "config_name": config_name,
            "dd2_batch_skip": dd2_batch_skip,
            "target_score": target_score,
            "max_iterations": max_iterations,
            "source_candidate_id": resolved_source_candidate_id,
            "instance_token": resolved_instance_token,
            "source_identity_summary": resolved_identity_summary,
            "baseline_dataset_dir": resolved_baseline_dataset,
            "baseline_output_dir": baseline_output_dir,
        },
        "source_binding": binding,
        "refinement_proposal": proposal,
        "generated_command": {
            "argv": command_args,
            "shell": shell_command(command_args) if command_args else None,
            "would_run_dd2_if_executed": bool(command_args),
            "requires_explicit_user_approval": True,
            "do_not_execute_automatically": True,
        },
        "planner_execution": {
            "does_not_run_gpu": True,
            "does_not_call_dd2": True,
            "does_not_generate_video": True,
            "does_not_mutate_outputs_except_plan_files": True,
        },
        "claim_boundary": {
            "command_plan_is_not_gpu_approval": True,
            "command_plan_is_not_video_semantic_success": True,
            "video_or_tensor_existence_is_not_semantic_success": True,
            "post_gpu_review_required_after_execution": True,
            "semantic_success_requires_measured_alignment_review": True,
        },
        "next_required_steps": [
            "human_reviews_command_draft",
            "explicit_user_approval_required_before_execution",
            "run_gpu_retry_only_after_approval",
            "run_post_gpu_review_pack",
            "attach_manual_or_perception_alignment_report",
            "run_prompt_video_alignment_eval",
        ],
    }


def render_markdown(plan: dict[str, Any]) -> str:
    command = as_dict(plan.get("generated_command"))
    resolved = as_dict(plan.get("resolved_inputs"))
    lines = [
        f"# Regeneration Command Plan: {plan.get('case_id')}",
        "",
        f"- Status: `{plan.get('command_status')}`",
        f"- Requires explicit approval: `{command.get('requires_explicit_user_approval')}`",
        f"- Planner runs GPU: `{not as_dict(plan.get('planner_execution')).get('does_not_run_gpu', True)}`",
        "",
        "## Resolved Inputs",
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    for key, value in resolved.items():
        lines.append(f"| `{key}` | `{value}` |")

    lines.extend(["", "## Command Draft", ""])
    if command.get("shell"):
        lines.extend(["```bash", str(command["shell"]), "```"])
    else:
        lines.append("Command draft unavailable because required fields are missing.")

    lines.extend(["", "## Claim Boundary", ""])
    for key, value in as_dict(plan.get("claim_boundary")).items():
        lines.append(f"- `{key}`: `{value}`")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an audit-only GPU regeneration command draft from a closed-loop trace.")
    parser.add_argument("--orchestrator-trace", required=True, type=Path)
    parser.add_argument("--source-binding-readiness", type=Path, default=None)
    parser.add_argument("--scenario-id", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--config-name", default="drivedreamer2_img_cond_mini_local")
    parser.add_argument("--dd2-batch-skip", type=int, default=0)
    parser.add_argument("--target-score", type=float, default=0.9)
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--source-candidate-id", default=None)
    parser.add_argument("--instance-token", default=None)
    parser.add_argument("--source-identity-summary", default=None)
    parser.add_argument("--baseline-dataset-dir", default=None)
    parser.add_argument("--baseline-output-dir", default=DEFAULT_BASELINE_OUTPUT_DIR)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    plan = build_regeneration_command_plan(
        orchestrator_trace=load_json(args.orchestrator_trace),
        source_binding_readiness=load_json(args.source_binding_readiness) if args.source_binding_readiness else None,
        scenario_id=args.scenario_id,
        output_dir=args.output_dir,
        config_name=args.config_name,
        dd2_batch_skip=args.dd2_batch_skip,
        target_score=args.target_score,
        max_iterations=args.max_iterations,
        source_candidate_id=args.source_candidate_id,
        instance_token=args.instance_token,
        source_identity_summary=args.source_identity_summary,
        baseline_dataset_dir=args.baseline_dataset_dir,
        baseline_output_dir=args.baseline_output_dir,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(args.output_json)

    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(plan), encoding="utf-8")
        print(args.output_md)

    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
