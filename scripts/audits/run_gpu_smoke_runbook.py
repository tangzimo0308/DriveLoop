from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_PLAN = Path("outputs/driveloop/gpu_smoke_command_plan/motorcycle_refined_candidate_plan.json")


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def code_block(command: str) -> str:
    return f"```bash\n{command}\n```"


def render_runbook(plan: dict[str, Any]) -> str:
    commands = plan.get("commands", {})
    claim_boundary = plan.get("claim_boundary", {})
    notes = plan.get("notes", [])

    scenario_id = plan.get("scenario_id", "unknown_scenario")
    prompt = plan.get("prompt", "")
    expected_video_path = plan.get("expected_video_path", "")

    lines = [
        "# GPU Smoke Runbook v0",
        "",
        "## Scope",
        "",
        "This runbook describes how to run one audited DriveDreamer-2 GPU smoke candidate.",
        "",
        f"- Scenario: `{scenario_id}`",
        f"- Expected video: `{expected_video_path}`",
        f"- Prompt: {prompt}",
        "",
        "## Claim Boundary",
        "",
        f"- GPU output claim: `{claim_boundary.get('gpu_smoke_generates', 'candidate_video_only')}`",
        f"- Semantic success allowed after GPU alone: `{claim_boundary.get('semantic_claim_allowed_after_gpu', False)}`",
        f"- Lane-change control claim allowed: `{claim_boundary.get('lane_change_control_claim_allowed', False)}`",
        f"- Required before semantic claim: {claim_boundary.get('required_before_semantic_claim', 'explicit review evidence')}",
        "",
        "Do not claim prompt-to-video semantic success from GPU generation alone.",
        "Do not claim visible lane change unless explicit manual, perception, or VLM review supports it.",
        "",
        "## Step 1: Readiness Gate",
        "",
        "Run this first. Continue only if the output reports `gpu_smoke_allowed: true`.",
        "",
        code_block(commands.get("readiness_gate", "")),
        "",
        "## Step 2: Candidate GPU Smoke",
        "",
        "Run this only after Step 1 passes. The result is still only a candidate video.",
        "",
        code_block(commands.get("gpu_smoke_candidate_generation", "")),
        "",
        "## Step 3: Post-GPU Review Gate",
        "",
        "Run this immediately after the candidate video exists. This keeps the video status as `not_measured` and creates the review pack.",
        "",
        code_block(commands.get("post_gpu_review_gate", "")),
        "",
        "## Step 4: Complete Review Report",
        "",
        "Manually inspect the review pack or attach perception/VLM evidence. Edit the generated manual alignment report with explicit pass/fail evidence.",
        "",
        "## Step 5: Alignment Evaluation",
        "",
        "Run this only after the explicit review report has been completed.",
        "",
        code_block(commands.get("alignment_eval_after_completed_review", "")),
        "",
        "## Notes",
        "",
    ]

    for note in notes:
        lines.append(f"- {note}")

    lines.extend(
        [
            "",
            "## Negative Result Policy",
            "",
            "If the generated candidate does not show the requested behavior, record `measured_failed`. Do not hide or re-label negative results.",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a human-readable GPU smoke runbook from a command plan.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    plan = load_json(args.plan)
    runbook = render_runbook(plan)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(runbook, encoding="utf-8")
    print(args.output)
    print(runbook)


if __name__ == "__main__":
    main()
