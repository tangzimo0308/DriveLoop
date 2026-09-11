from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_RUNTIME_COMPARE = Path("outputs/driveloop/dd2_runtime_hash_compare/motorcycle_earlier_vs_refined.json")
DEFAULT_MOTION_GAP = Path("outputs/driveloop/motion_control_gap_audit/motorcycle_manual_feedback_motion_gap.json")
DEFAULT_VELOCITY_SURFACE = Path("outputs/driveloop/dd2_velocity_surface_audit/mini_velocity_surface.json")
DEFAULT_TRAJECTORY_CONTRACT_DOC = Path("experiments/2026-06-28_trajectory_control_contract_v0.md")
DEFAULT_CONFIG = Path("dreamer-train/projects/DriveDreamer2/configs/drivedreamer2_img_cond_mini_local.py")
DEFAULT_LABELS = Path("/data/projects/DriveLoop/data/processed/nuscenes/v1.0-mini/cam_all_val/v0.0.2/labels/data.pkl")
DEFAULT_WEIGHTS = Path("/data/projects/DriveLoop/pretrained_models/drivedreamer2_img_cond/pytorch_gligen_weights.bin")


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def evidence_item(path: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "description": description,
        "exists": path.exists(),
    }


def build_readiness_report(
    prompt: str,
    scenario_id: str,
    runtime_compare: Path = DEFAULT_RUNTIME_COMPARE,
    motion_gap: Path = DEFAULT_MOTION_GAP,
    velocity_surface: Path = DEFAULT_VELOCITY_SURFACE,
    trajectory_contract_doc: Path = DEFAULT_TRAJECTORY_CONTRACT_DOC,
    config_path: Path = DEFAULT_CONFIG,
    labels_path: Path = DEFAULT_LABELS,
    weights_path: Path = DEFAULT_WEIGHTS,
) -> dict[str, Any]:
    runtime_data = load_json(runtime_compare)
    motion_gap_data = load_json(motion_gap)
    velocity_data = load_json(velocity_surface)

    runtime_changed = runtime_data.get("runtime_tensor_hash_changed", {})
    motion_claim = motion_gap_data.get("claim", {})
    if not isinstance(motion_claim, dict):
        motion_claim = {}
    motion_control_status = motion_gap_data.get("control_path_status", {})
    if not isinstance(motion_control_status, dict):
        motion_control_status = {}
    velocity_claim = velocity_data.get("claim", {})

    required_evidence = {
        "runtime_hash_compare": evidence_item(
            runtime_compare,
            "Confirms which DD2 runtime hashes changed for earlier vs refined audit-only.",
        ),
        "motion_control_gap_audit": evidence_item(
            motion_gap,
            "Records that lane-change temporal motion tensor control is not verified.",
        ),
        "velocity_surface_audit": evidence_item(
            velocity_surface,
            "Records that dataset velocities exist but are not consumed by DD2 runtime input.",
        ),
        "trajectory_control_contract": evidence_item(
            trajectory_contract_doc,
            "Defines evidence required before claiming trajectory control.",
        ),
    }
    runtime_resources = {
        "dd2_config": evidence_item(config_path, "DD2 local mini config."),
        "mini_labels": evidence_item(labels_path, "Processed mini labels for DD2 input."),
        "dd2_weights": evidence_item(weights_path, "DD2 image-conditioned weights."),
    }

    evidence_ready = all(item["exists"] for item in required_evidence.values())
    resources_ready = all(item["exists"] for item in runtime_resources.values())

    expected_runtime_boundary_ok = (
        runtime_changed.get("prompt_embed") is True
        and runtime_changed.get("box_downsampler_input") is False
        and runtime_changed.get("grounding_downsampler_input") is False
        and runtime_changed.get("img_cond") is False
    )
    motion_gap_ok = (
        motion_claim.get("lane_change_motion_tensor_control") == "not_verified"
        and motion_claim.get("semantic_lane_change_claim") == "not_allowed"
        and motion_claim.get("semantic_success_claim_allowed") is False
        and motion_control_status.get("trajectory_tensor") == "not_implemented"
        and motion_control_status.get("semantic_lane_change_claim") == "not_allowed"
    )
    velocity_gap_ok = velocity_claim.get("velocity_consumed_by_dd2_runtime") is False

    gpu_smoke_allowed = bool(
        evidence_ready
        and resources_ready
        and expected_runtime_boundary_ok
        and motion_gap_ok
        and velocity_gap_ok
    )

    return {
        "schema_version": "driveloop_gpu_smoke_readiness_gate.v0",
        "scenario_id": scenario_id,
        "prompt": prompt,
        "gpu_smoke_allowed": gpu_smoke_allowed,
        "semantic_claim_allowed": False,
        "allowed_claim_after_gpu": "candidate_video_generated_only",
        "required_followup_after_gpu": [
            "preserve video artifact",
            "preserve dd2_runtime_input_audit",
            "generate manual review pack",
            "run prompt-video alignment evaluation from explicit review report",
        ],
        "required_evidence": required_evidence,
        "runtime_resources": runtime_resources,
        "evidence_checks": {
            "evidence_ready": evidence_ready,
            "resources_ready": resources_ready,
            "runtime_boundary_ok": expected_runtime_boundary_ok,
            "motion_gap_recorded": motion_gap_ok,
            "motion_gap_semantic_lane_change_claim": motion_claim.get("semantic_lane_change_claim"),
            "motion_gap_semantic_success_claim_allowed": motion_claim.get("semantic_success_claim_allowed"),
            "motion_gap_control_path_status": motion_control_status,
            "velocity_gap_recorded": velocity_gap_ok,
        },
        "claim_boundary": (
            "Passing this gate only allows a short GPU smoke to generate a candidate video. "
            "It does not allow a prompt-to-video semantic success claim, and it does not prove lane-change control."
        ),
        "recommended_command_note": (
            "Run a single short DD2 smoke only if gpu_smoke_allowed is true; treat the result as not_measured "
            "until manual/perception/VLM review is attached."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether a short DD2 GPU smoke is audit-ready.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--runtime-compare", type=Path, default=DEFAULT_RUNTIME_COMPARE)
    parser.add_argument("--motion-gap", type=Path, default=DEFAULT_MOTION_GAP)
    parser.add_argument("--velocity-surface", type=Path, default=DEFAULT_VELOCITY_SURFACE)
    parser.add_argument("--trajectory-contract-doc", type=Path, default=DEFAULT_TRAJECTORY_CONTRACT_DOC)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    report = build_readiness_report(
        prompt=args.prompt,
        scenario_id=args.scenario_id,
        runtime_compare=args.runtime_compare,
        motion_gap=args.motion_gap,
        velocity_surface=args.velocity_surface,
        trajectory_contract_doc=args.trajectory_contract_doc,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
