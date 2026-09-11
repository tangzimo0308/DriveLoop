from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_READINESS_GATE = Path("outputs/driveloop/gpu_smoke_readiness/candidate70_gpu_readiness_gate.json")
DEFAULT_TEMPLATE_OUTPUT = Path("outputs/driveloop/gpu_retry_approval/candidate70_gpu_retry_approval.template.json")
DEFAULT_APPROVAL_TARGET = Path("outputs/driveloop/gpu_retry_approval/candidate70_gpu_retry_approval.json")
SCENARIO_ID = "candidate70_night_cut_in_gpu_smoke"


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _ready_except_approval(retry_gate: dict[str, Any]) -> bool:
    checks = retry_gate.get("checks", {})
    if not isinstance(checks, dict):
        checks = {}
    required = [
        "source_bound_actor_motion_full_coverage_verified",
        "true_lane_geometry_replacement_available",
        "semantic_alignment_protocol_defined",
        "closed_loop_status_has_perception_measured_failed",
        "perception_eval_measured_failed",
        "semantic_success_claim_allowed_remains_false",
    ]
    return all(checks.get(key) is True for key in required)


def build_approval_template(
    readiness_gate: dict[str, Any],
    *,
    readiness_gate_path: Path = DEFAULT_READINESS_GATE,
    approval_target_path: Path = DEFAULT_APPROVAL_TARGET,
    approve: bool = False,
    approved_by: str | None = None,
    approval_note: str | None = None,
) -> dict[str, Any]:
    retry_gate = readiness_gate.get("gpu_retry_gate", {})
    if not isinstance(retry_gate, dict):
        retry_gate = {}

    checks = retry_gate.get("checks", {})
    if not isinstance(checks, dict):
        checks = {}

    ready_except_approval = _ready_except_approval(retry_gate)
    can_approve = (
        readiness_gate.get("scenario_id") == SCENARIO_ID
        and retry_gate.get("status") == "blocked_requires_explicit_user_approval"
        and retry_gate.get("requires_post_gpu_review") is True
        and retry_gate.get("does_not_claim_semantic_success") is True
        and ready_except_approval
    )
    approved = bool(approve and can_approve)

    approval_blockers = []
    if not ready_except_approval:
        approval_blockers.append("non_gpu_retry_preconditions_not_satisfied")
    if not approve:
        approval_blockers.append("explicit_user_approval_not_recorded")
    if approve and not approved:
        approval_blockers.append("approval_rejected_by_gate_preconditions")

    return {
        "schema_version": "driveloop_candidate70_gpu_retry_approval.v0",
        "scenario_id": SCENARIO_ID,
        "approval_status": "approved_for_one_short_gpu_retry" if approved else "template_not_approved",
        "approved_for_candidate70_gpu_retry": approved,
        "approved_by": approved_by,
        "approval_note": approval_note or "template only; explicit user approval not recorded",
        "approval_target_path": str(approval_target_path),
        "requires_post_gpu_review": True,
        "approval_is_not_semantic_success": True,
        "does_not_run_gpu": True,
        "does_not_generate_video": True,
        "preconditions": {
            "ready_except_explicit_approval": ready_except_approval,
            "readiness_gate_status": readiness_gate.get("readiness_status"),
            "gpu_retry_gate_status": retry_gate.get("status"),
            "gpu_retry_gate_blockers": retry_gate.get("blockers", []),
            "checks": checks,
        },
        "approval_blockers": approval_blockers,
        "sources": {
            "candidate70_gpu_readiness_gate": {
                "path": str(readiness_gate_path),
                "exists": readiness_gate_path.exists(),
            }
        },
        "claim_boundary": {
            "approval_template_is_not_gpu_execution": True,
            "approval_is_not_video_semantic_success": True,
            "approval_requires_post_gpu_review": True,
            "semantic_success_claim_allowed_remains_false_until_measured_passed_review": True,
        },
        "next_required_steps_after_approval": [
            "rerun candidate70 gpu readiness gate and confirm gpu_retry_gate.allowed is true",
            "run at most one short GPU retry only after explicit user approval",
            "immediately run post-GPU review gate after the candidate video exists",
            "record measured_failed or measured_passed from explicit review evidence",
        ],
    }


def write_template(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a candidate70 GPU retry approval template.")
    parser.add_argument("--readiness-gate", type=Path, default=DEFAULT_READINESS_GATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_TEMPLATE_OUTPUT)
    parser.add_argument("--approval-target", type=Path, default=DEFAULT_APPROVAL_TARGET)
    parser.add_argument("--approve", action="store_true")
    parser.add_argument("--approved-by", default=None)
    parser.add_argument("--approval-note", default=None)
    args = parser.parse_args()

    if args.approve and (not args.approved_by or not args.approval_note):
        raise SystemExit("--approve requires --approved-by and --approval-note")

    payload = build_approval_template(
        load_json(args.readiness_gate),
        readiness_gate_path=args.readiness_gate,
        approval_target_path=args.approval_target,
        approve=args.approve,
        approved_by=args.approved_by,
        approval_note=args.approval_note,
    )
    write_template(args.output, payload)
    print(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
