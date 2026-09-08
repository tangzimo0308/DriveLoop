from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def path_or_none(value: str | None) -> Path | None:
    return Path(value) if value else None


def add_path_arg(command: list[str], flag: str, value: Path | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def add_str_arg(command: list[str], flag: str, value: str | None) -> None:
    if value:
        command.extend([flag, value])


def add_float_arg(command: list[str], flag: str, value: float | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def add_int_arg(command: list[str], flag: str, value: int | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def artifact_paths(output_root: Path) -> dict[str, Path]:
    return {
        "orchestrator_json": output_root / "closed_loop_trace.json",
        "orchestrator_md": output_root / "closed_loop_trace.md",
        "regeneration_plan_json": output_root / "regeneration_command_plan.json",
        "regeneration_plan_md": output_root / "regeneration_command_plan.md",
        "post_regeneration_review_plan_json": output_root / "post_regeneration_review_plan.json",
        "post_regeneration_review_plan_md": output_root / "post_regeneration_review_plan.md",
        "case_summary_json": output_root / "closed_loop_case_summary.json",
        "case_summary_md": output_root / "closed_loop_case_summary.md",
        "runner_summary_json": output_root / "runner_summary.json",
        "runner_summary_md": output_root / "runner_summary.md",
    }


def build_commands(args: argparse.Namespace, paths: dict[str, Path]) -> list[dict[str, Any]]:
    py = sys.executable
    commands: list[dict[str, Any]] = []

    orchestrator = [
        py,
        "scripts/run_audit_only_closed_loop_orchestrator.py",
        "--case-id",
        args.case_id,
        "--initial-alignment-eval",
        str(args.initial_alignment_eval),
        "--target-score",
        str(args.target_score),
        "--max-iterations",
        str(args.max_iterations),
        "--output-json",
        str(paths["orchestrator_json"]),
        "--output-md",
        str(paths["orchestrator_md"]),
    ]
    add_path_arg(orchestrator, "--retry-alignment-eval", args.retry_alignment_eval)
    add_path_arg(orchestrator, "--failure-taxonomy", args.failure_taxonomy)
    add_path_arg(orchestrator, "--refinement-proposal", args.refinement_proposal)
    add_path_arg(orchestrator, "--candidate-audit", args.candidate_audit)
    commands.append(
        {
            "step": "closed_loop_orchestrator",
            "description": "Build audit-only closed-loop trace.",
            "argv": orchestrator,
            "writes": [str(paths["orchestrator_json"]), str(paths["orchestrator_md"])],
        }
    )

    regeneration = [
        py,
        "scripts/run_audit_only_regeneration_command_planner.py",
        "--orchestrator-trace",
        str(paths["orchestrator_json"]),
        "--scenario-id",
        args.scenario_id,
        "--output-dir",
        str(args.regeneration_output_dir),
        "--config-name",
        args.config_name,
        "--dd2-batch-skip",
        str(args.dd2_batch_skip),
        "--target-score",
        str(args.target_score),
        "--max-iterations",
        str(args.max_iterations),
        "--output-json",
        str(paths["regeneration_plan_json"]),
        "--output-md",
        str(paths["regeneration_plan_md"]),
    ]
    add_path_arg(regeneration, "--source-binding-readiness", args.source_binding_readiness)
    add_str_arg(regeneration, "--source-candidate-id", args.source_candidate_id)
    add_str_arg(regeneration, "--instance-token", args.instance_token)
    add_path_arg(regeneration, "--source-identity-summary", args.source_identity_summary)
    add_path_arg(regeneration, "--baseline-dataset-dir", args.baseline_dataset_dir)
    add_path_arg(regeneration, "--baseline-output-dir", args.baseline_output_dir)
    commands.append(
        {
            "step": "regeneration_command_planner",
            "description": "Draft the next GPU regeneration command without executing it.",
            "argv": regeneration,
            "writes": [str(paths["regeneration_plan_json"]), str(paths["regeneration_plan_md"])],
        }
    )

    review = [
        py,
        "scripts/run_post_regeneration_review_planner.py",
        "--regeneration-command-plan",
        str(paths["regeneration_plan_json"]),
        "--pass-threshold",
        str(args.target_score),
        "--output-json",
        str(paths["post_regeneration_review_plan_json"]),
        "--output-md",
        str(paths["post_regeneration_review_plan_md"]),
        "--summary-output-json",
        str(paths["case_summary_json"]),
        "--summary-output-md",
        str(paths["case_summary_md"]),
    ]
    add_path_arg(review, "--video-path", args.video_path)
    add_path_arg(review, "--post-gpu-output-dir", args.post_gpu_output_dir)
    add_path_arg(review, "--alignment-eval-output-dir", args.alignment_eval_output_dir)
    add_path_arg(review, "--failed-alignment-eval", args.initial_alignment_eval)
    add_path_arg(review, "--failure-taxonomy", args.failure_taxonomy)
    add_path_arg(review, "--failed-perception-eval", args.failed_perception_eval)
    add_path_arg(review, "--refinement-proposal", args.refinement_proposal)
    commands.append(
        {
            "step": "post_regeneration_review_planner",
            "description": "Plan manual/perception review commands for a future regenerated video.",
            "argv": review,
            "writes": [
                str(paths["post_regeneration_review_plan_json"]),
                str(paths["post_regeneration_review_plan_md"]),
            ],
        }
    )

    if args.retry_alignment_eval is not None:
        summary = [
            py,
            "scripts/run_closed_loop_case_summary.py",
            "--case-id",
            args.case_id,
            "--failed-alignment-eval",
            str(args.initial_alignment_eval),
            "--retry-alignment-eval",
            str(args.retry_alignment_eval),
            "--output-json",
            str(paths["case_summary_json"]),
            "--output-md",
            str(paths["case_summary_md"]),
        ]
        add_path_arg(summary, "--failure-taxonomy", args.failure_taxonomy)
        add_path_arg(summary, "--failed-perception-eval", args.failed_perception_eval)
        add_path_arg(summary, "--refinement-proposal", args.refinement_proposal)
        commands.append(
            {
                "step": "closed_loop_case_summary",
                "description": "Summarize failed-to-passed evidence when retry evaluation is available.",
                "argv": summary,
                "writes": [str(paths["case_summary_json"]), str(paths["case_summary_md"])],
            }
        )

    return commands


def tail_text(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def execute_command(command: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    completed = subprocess.run(
        command["argv"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.stderr:
        print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n", file=sys.stderr)
    return {
        "step": command["step"],
        "returncode": completed.returncode,
        "stdout_tail": tail_text(completed.stdout),
        "stderr_tail": tail_text(completed.stderr),
        "writes": command["writes"],
    }


def infer_status(results: list[dict[str, Any]], commands: list[dict[str, Any]], retry_alignment_eval: Path | None) -> str:
    if len(results) != len(commands):
        return "failed_before_all_steps_completed"
    if any(result["returncode"] != 0 for result in results):
        return "failed"
    if retry_alignment_eval is not None:
        return "audit_only_completed_with_retry_evidence"
    return "audit_only_completed_pending_regeneration"


def build_summary(
    args: argparse.Namespace,
    paths: dict[str, Path],
    commands: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    generated = {name: str(path) for name, path in paths.items()}
    existing = {name: path.exists() for name, path in paths.items()}
    status = infer_status(results, commands, args.retry_alignment_eval)

    return {
        "schema_version": "driveloop_audit_only_closed_loop_runner.v0",
        "case_id": args.case_id,
        "runner_status": status,
        "target_score": args.target_score,
        "max_iterations": args.max_iterations,
        "inputs": {
            "initial_alignment_eval": str(args.initial_alignment_eval),
            "retry_alignment_eval": str(args.retry_alignment_eval) if args.retry_alignment_eval else None,
            "failure_taxonomy": str(args.failure_taxonomy) if args.failure_taxonomy else None,
            "failed_perception_eval": str(args.failed_perception_eval) if args.failed_perception_eval else None,
            "refinement_proposal": str(args.refinement_proposal) if args.refinement_proposal else None,
            "source_binding_readiness": str(args.source_binding_readiness) if args.source_binding_readiness else None,
        },
        "generated_artifacts": generated,
        "artifact_exists": existing,
        "planned_commands": commands,
        "executed_results": results,
        "runner_execution": {
            "ran_gpu": False,
            "called_dd2_runtime": False,
            "generated_video": False,
            "trained_model": False,
            "inspected_video_pixels": False,
            "made_semantic_success_claim_from_video_existence": False,
        },
        "claim_boundary": {
            "runner_is_audit_only": True,
            "runner_does_not_grant_gpu_approval": True,
            "runner_does_not_execute_regeneration_command": True,
            "runner_does_not_make_video_semantic_success_claim": True,
            "semantic_success_claim_allowed": False,
        },
        "next_required_steps": [
            "If regeneration is needed, obtain explicit user approval before running the generated GPU command.",
            "After a regenerated video exists, run the planned post-regeneration review commands.",
            "Only record measured_passed after manual/perception evidence supports the semantic claim.",
        ],
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Audit-only closed-loop runner",
        "",
        f"- case_id: `{summary['case_id']}`",
        f"- runner_status: `{summary['runner_status']}`",
        f"- target_score: `{summary['target_score']}`",
        f"- semantic_success_claim_allowed: `{summary['claim_boundary']['semantic_success_claim_allowed']}`",
        "",
        "## Steps",
    ]
    for result in summary["executed_results"]:
        lines.append(f"- `{result['step']}` returncode `{result['returncode']}`")
    lines.extend(["", "## Generated artifacts"])
    for name, artifact_path in summary["generated_artifacts"].items():
        exists = summary["artifact_exists"][name]
        lines.append(f"- `{name}`: `{artifact_path}` exists `{exists}`")
    lines.extend(["", "## Boundary"])
    for key, value in summary["runner_execution"].items():
        lines.append(f"- `{key}`: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path.cwd()
    paths = artifact_paths(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    commands = build_commands(args, paths)

    results: list[dict[str, Any]] = []
    for command in commands:
        print(f"===== {command['step']} =====")
        result = execute_command(command, repo_root)
        results.append(result)
        if result["returncode"] != 0:
            break

    summary = build_summary(args, paths, commands, results)
    write_json(paths["runner_summary_json"], summary)
    write_markdown(paths["runner_summary_md"], summary)
    print(f"wrote {paths['runner_summary_json']}")
    print(f"wrote {paths['runner_summary_md']}")

    if any(result["returncode"] != 0 for result in results):
        raise SystemExit(1)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the DriveLoop closed-loop audit chain without GPU execution."
    )
    parser.add_argument("--case-id", default="audit_only_closed_loop_case")
    parser.add_argument("--initial-alignment-eval", type=Path, required=True)
    parser.add_argument("--retry-alignment-eval", type=Path)
    parser.add_argument("--failure-taxonomy", type=Path)
    parser.add_argument("--failed-perception-eval", type=Path)
    parser.add_argument("--refinement-proposal", type=Path)
    parser.add_argument("--candidate-audit", type=Path)
    parser.add_argument("--source-binding-readiness", type=Path)
    parser.add_argument("--scenario-id", default="audit_only_next_gpu_retry_draft")
    parser.add_argument("--regeneration-output-dir", type=Path, default=Path("outputs/driveloop/audit_only_next_gpu_retry_draft"))
    parser.add_argument("--config-name", default="drivedreamer2_img_cond_mini_local")
    parser.add_argument("--dd2-batch-skip", type=int, default=0)
    parser.add_argument("--target-score", type=float, default=0.9)
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--source-candidate-id")
    parser.add_argument("--instance-token")
    parser.add_argument("--source-identity-summary", type=Path)
    parser.add_argument("--baseline-dataset-dir", type=Path)
    parser.add_argument("--baseline-output-dir", type=Path)
    parser.add_argument("--video-path", type=Path)
    parser.add_argument("--post-gpu-output-dir", type=Path)
    parser.add_argument("--alignment-eval-output-dir", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
