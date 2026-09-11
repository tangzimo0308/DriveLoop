from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def shell_command(args: list[str]) -> str:
    return " \\\n  ".join(shlex.quote(str(arg)) for arg in args)


def command_entry(name: str, argv: list[str], can_run_now: bool, blocked_until: str | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "argv": argv,
        "shell": shell_command(argv),
        "can_run_now": can_run_now,
        "blocked_until": blocked_until,
        "do_not_execute_automatically": True,
    }


def resolved_inputs_from_command_plan(command_plan: dict[str, Any]) -> dict[str, Any]:
    resolved = as_dict(command_plan.get("resolved_inputs"))
    generated_command = as_dict(command_plan.get("generated_command"))
    return {
        "prompt": resolved.get("prompt"),
        "scenario_id": resolved.get("scenario_id"),
        "generation_output_dir": resolved.get("output_dir"),
        "regeneration_command_status": command_plan.get("command_status"),
        "regeneration_requires_explicit_user_approval": generated_command.get("requires_explicit_user_approval"),
    }


def default_video_path(generation_output_dir: str, scenario_id: str) -> Path:
    return Path(generation_output_dir) / "artifacts" / scenario_id / "iteration_00.mp4"


def build_post_regeneration_review_plan(
    regeneration_command_plan: dict[str, Any],
    *,
    video_path: str | None = None,
    post_gpu_output_dir: str | None = None,
    alignment_eval_output_dir: str = "outputs/driveloop/prompt_video_alignment_eval",
    pass_threshold: float = 0.8,
    failed_alignment_eval: str | None = None,
    failure_taxonomy: str | None = None,
    failed_perception_eval: str | None = None,
    refinement_proposal: str | None = None,
    summary_output_json: str | None = None,
    summary_output_md: str | None = None,
) -> dict[str, Any]:
    inputs = resolved_inputs_from_command_plan(regeneration_command_plan)
    prompt = inputs.get("prompt")
    scenario_id = inputs.get("scenario_id")
    generation_output_dir = inputs.get("generation_output_dir")

    missing_required = []
    if not prompt:
        missing_required.append("prompt")
    if not scenario_id:
        missing_required.append("scenario_id")
    if not generation_output_dir and not video_path:
        missing_required.append("generation_output_dir_or_video_path")

    if scenario_id and generation_output_dir:
        resolved_video_path = Path(video_path) if video_path else default_video_path(str(generation_output_dir), str(scenario_id))
    elif video_path:
        resolved_video_path = Path(video_path)
    else:
        resolved_video_path = Path("missing_video_path")

    resolved_post_gpu_dir = Path(post_gpu_output_dir) if post_gpu_output_dir else Path("outputs/driveloop/post_gpu_review_gate") / str(scenario_id or "unknown_scenario")
    manual_pack_dir = resolved_post_gpu_dir / "manual_review_pack"
    manual_report_template = manual_pack_dir / "manual_alignment_report_template.json"
    manual_report = manual_pack_dir / "manual_alignment_report.json"
    alignment_eval_dir = Path(alignment_eval_output_dir)
    alignment_eval_path = alignment_eval_dir / str(scenario_id or "unknown_scenario") / "prompt_video_alignment_evaluation.json"

    post_gate_args: list[str] = []
    alignment_eval_args: list[str] = []
    summary_args: list[str] = []

    if not missing_required:
        post_gate_args = [
            "python",
            "scripts/run_post_gpu_review_gate.py",
            "--prompt",
            str(prompt),
            "--scenario-id",
            str(scenario_id),
            "--video-path",
            str(resolved_video_path),
            "--output-dir",
            str(resolved_post_gpu_dir),
            "--pass-threshold",
            str(pass_threshold),
        ]

        alignment_eval_args = [
            "python",
            "scripts/run_prompt_video_alignment_eval.py",
            "--prompt",
            str(prompt),
            "--scenario-id",
            str(scenario_id),
            "--video-path",
            str(resolved_video_path),
            "--alignment-report",
            str(manual_report),
            "--output-dir",
            str(alignment_eval_dir),
            "--pass-threshold",
            str(pass_threshold),
        ]

        if failed_alignment_eval:
            summary_output_json = summary_output_json or str(Path("outputs/driveloop/closed_loop_case_summary") / f"{scenario_id}_summary.json")
            summary_output_md = summary_output_md or str(Path("outputs/driveloop/closed_loop_case_summary") / f"{scenario_id}_summary.md")
            summary_args = [
                "python",
                "scripts/run_closed_loop_case_summary.py",
                "--case-id",
                str(scenario_id),
                "--failed-alignment-eval",
                str(failed_alignment_eval),
                "--retry-alignment-eval",
                str(alignment_eval_path),
                "--output-json",
                str(summary_output_json),
                "--output-md",
                str(summary_output_md),
            ]
            optional_pairs = [
                ("--failure-taxonomy", failure_taxonomy),
                ("--failed-perception-eval", failed_perception_eval),
                ("--refinement-proposal", refinement_proposal),
            ]
            for flag, value in optional_pairs:
                if value:
                    summary_args.extend([flag, str(value)])

    video_exists = resolved_video_path.exists()
    manual_report_exists = manual_report.exists()
    alignment_eval_exists = alignment_eval_path.exists()

    commands = {
        "post_gpu_review_gate": command_entry(
            "post_gpu_review_gate",
            post_gate_args,
            can_run_now=bool(post_gate_args and video_exists),
            blocked_until=None if video_exists else "generated_video_exists",
        ) if post_gate_args else None,
        "prompt_video_alignment_eval_after_completed_review": command_entry(
            "prompt_video_alignment_eval_after_completed_review",
            alignment_eval_args,
            can_run_now=bool(alignment_eval_args and video_exists and manual_report_exists),
            blocked_until=None if video_exists and manual_report_exists else "generated_video_and_completed_manual_report_exist",
        ) if alignment_eval_args else None,
        "closed_loop_summary_refresh_after_alignment_eval": command_entry(
            "closed_loop_summary_refresh_after_alignment_eval",
            summary_args,
            can_run_now=bool(summary_args and alignment_eval_exists),
            blocked_until=None if alignment_eval_exists else "retry_prompt_video_alignment_eval_exists",
        ) if summary_args else None,
    }

    plan_status = "draft_incomplete_missing_required_fields" if missing_required else "draft_ready_for_post_regeneration_review"

    return {
        "schema_version": "driveloop_post_regeneration_review_plan.v0",
        "plan_status": plan_status,
        "missing_required_fields": missing_required,
        "case_id": scenario_id,
        "resolved_inputs": {
            **inputs,
            "video_path": str(resolved_video_path),
            "post_gpu_output_dir": str(resolved_post_gpu_dir),
            "manual_review_pack_dir": str(manual_pack_dir),
            "manual_report_template": str(manual_report_template),
            "completed_manual_report_expected": str(manual_report),
            "alignment_eval_output_dir": str(alignment_eval_dir),
            "alignment_eval_path": str(alignment_eval_path),
            "pass_threshold": pass_threshold,
        },
        "artifact_status": {
            "video_exists": video_exists,
            "manual_report_exists": manual_report_exists,
            "alignment_eval_exists": alignment_eval_exists,
        },
        "commands": commands,
        "planner_execution": {
            "does_not_run_gpu": True,
            "does_not_call_dd2": True,
            "does_not_generate_video": True,
            "does_not_inspect_video_pixels": True,
            "does_not_make_semantic_success_claim": True,
        },
        "claim_boundary": {
            "review_plan_is_not_gpu_approval": True,
            "review_plan_is_not_video_semantic_success": True,
            "manual_report_template_is_not_measured_review": True,
            "prompt_video_alignment_eval_requires_completed_external_report": True,
            "semantic_success_requires_measured_alignment_review": True,
        },
        "next_required_steps": [
            "wait_for_or_verify_generated_video_artifact",
            "run_post_gpu_review_gate_to_create_review_pack",
            "human_or_perception_system_completes_manual_alignment_report",
            "run_prompt_video_alignment_eval_with_completed_report",
            "refresh_closed_loop_summary",
        ],
    }


def render_markdown(plan: dict[str, Any]) -> str:
    lines = [
        f"# Post-regeneration Review Plan: {plan.get('case_id')}",
        "",
        f"- Status: `{plan.get('plan_status')}`",
        "",
        "## Artifact Status",
        "",
    ]
    for key, value in as_dict(plan.get("artifact_status")).items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(["", "## Commands", ""])
    commands = as_dict(plan.get("commands"))
    for name, command in commands.items():
        if not command:
            continue
        command = as_dict(command)
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Can run now: `{command.get('can_run_now')}`",
                f"- Blocked until: `{command.get('blocked_until')}`",
                "",
                "```bash",
                str(command.get("shell")),
                "```",
                "",
            ]
        )

    lines.extend(["## Claim Boundary", ""])
    for key, value in as_dict(plan.get("claim_boundary")).items():
        lines.append(f"- `{key}`: `{value}`")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan post-regeneration DriveLoop review/evaluation commands without executing them.")
    parser.add_argument("--regeneration-command-plan", required=True, type=Path)
    parser.add_argument("--video-path", default=None)
    parser.add_argument("--post-gpu-output-dir", default=None)
    parser.add_argument("--alignment-eval-output-dir", default="outputs/driveloop/prompt_video_alignment_eval")
    parser.add_argument("--pass-threshold", type=float, default=0.8)
    parser.add_argument("--failed-alignment-eval", default=None)
    parser.add_argument("--failure-taxonomy", default=None)
    parser.add_argument("--failed-perception-eval", default=None)
    parser.add_argument("--refinement-proposal", default=None)
    parser.add_argument("--summary-output-json", default=None)
    parser.add_argument("--summary-output-md", default=None)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    plan = build_post_regeneration_review_plan(
        regeneration_command_plan=load_json(args.regeneration_command_plan),
        video_path=args.video_path,
        post_gpu_output_dir=args.post_gpu_output_dir,
        alignment_eval_output_dir=args.alignment_eval_output_dir,
        pass_threshold=args.pass_threshold,
        failed_alignment_eval=args.failed_alignment_eval,
        failure_taxonomy=args.failure_taxonomy,
        failed_perception_eval=args.failed_perception_eval,
        refinement_proposal=args.refinement_proposal,
        summary_output_json=args.summary_output_json,
        summary_output_md=args.summary_output_md,
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
