from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Dict, List

from driveloop import DriveLoopRequest
from driveloop.condition_adapter import DriveDreamer2ConditionAdapter
from driveloop.grounding import RuleBasedGrounder
from driveloop.longtail import LongTailController
from driveloop.refiner import RuleBasedRefiner
from driveloop.schema import Diagnosis, Evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare an audit-only refined DD2 condition from an alignment report."
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--alignment-report", required=True)
    parser.add_argument("--scenario-id", default="alignment_feedback_audit_only")
    parser.add_argument("--output-dir", default="outputs/driveloop/alignment_feedback_audit_only")
    return parser.parse_args()


def load_failed_checks(path: str | Path) -> List[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("alignment report must be a JSON object")

    report = data
    for key in ("prompt_video_alignment", "video_alignment_report", "perception_alignment"):
        value = data.get(key)
        if isinstance(value, dict):
            report = value
            break

    checks = report.get("checks", [])
    if not isinstance(checks, list):
        raise ValueError("alignment report checks must be a list")

    failed: List[str] = []
    for check in checks:
        if not isinstance(check, dict):
            continue
        if bool(check.get("required", True)) and not bool(check.get("passed", False)):
            name = check.get("name")
            if isinstance(name, str):
                failed.append(name)
    return failed


def build_alignment_feedback_request(prompt: str, failed_checks: List[str]) -> DriveLoopRequest:
    evaluation = Evaluation(
        score=0.0,
        diagnosis=Diagnosis(
            passed=False,
            reasons=[f"alignment_check_failed:{check}" for check in failed_checks],
            suggested_actions=["inspect failed alignment checks before making semantic claims"],
        ),
    )
    refinement = RuleBasedRefiner().refine(DriveLoopRequest(prompt=prompt), evaluation)
    return DriveLoopRequest(
        prompt=refinement.prompt,
        condition=refinement.condition,
        metadata={
            "alignment_feedback_source": "manual_review",
            "alignment_feedback_failed_checks": failed_checks,
            "refinement_notes": refinement.notes,
        },
    )


def prepare_summary(prompt: str, alignment_report: str | Path, scenario_id: str) -> Dict[str, Any]:
    failed_checks = load_failed_checks(alignment_report)
    request = build_alignment_feedback_request(prompt, failed_checks)

    grounder = RuleBasedGrounder()
    longtail = LongTailController()
    adapter = DriveDreamer2ConditionAdapter()

    scene_spec = grounder.ground(request)
    condition_plan = longtail.build(scene_spec)
    dd2_condition = adapter.build(
        scene_spec,
        condition_plan,
        alignment_feedback=request.condition.get("alignment_feedback"),
    )
    executable_condition = dd2_condition.executable_condition
    trace = executable_condition.get("trace_metadata", {})

    return {
        "scenario_id": scenario_id,
        "source_prompt": prompt,
        "refined_prompt": request.prompt,
        "failed_checks": failed_checks,
        "scene_specification": asdict(scene_spec),
        "dd2_condition": asdict(dd2_condition),
        "audit_summary": {
            "tensor_control_claim": "not_evaluated",
            "video_semantic_claim": "not_evaluated",
            "alignment_feedback_trace_present": "alignment_feedback" in trace,
            "alignment_feedback": trace.get("alignment_feedback"),
            "motion_controls": executable_condition.get("motion_controls", []),
            "actor_controls": executable_condition.get("actor_controls", []),
            "claim_boundary": (
                "This script prepares a refined DD2 condition for audit only. "
                "It does not run DD2 diffusion, inspect video pixels, or prove tensor-level correction."
            ),
        },
    }


def main() -> None:
    args = parse_args()
    payload = prepare_summary(args.prompt, args.alignment_report, args.scenario_id)

    output_dir = Path(args.output_dir) / args.scenario_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "alignment_feedback_audit_only_summary.json"
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Wrote alignment feedback audit-only summary: {output_path}")


if __name__ == "__main__":
    main()
