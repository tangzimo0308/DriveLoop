from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.run_prompt_conditional_candidate_audit import RULES, contains_any


DEFAULT_PROMPT = "daytime urban road with a motorcycle, the motorcycle performs a visible lane change from the left, panoramic multi-view video."
DEFAULT_BACKEND_SUMMARY = Path(
    "outputs/driveloop/motorcycle_manual_feedback_dd2_audit_only/"
    "motorcycle_manual_feedback_dd2_audit_only/backend_audit_only_summary.json"
)
DEFAULT_VELOCITY_AUDIT = Path("outputs/driveloop/dd2_velocity_surface_audit/mini_velocity_surface.json")
DEFAULT_MOTION_GAP = Path("outputs/driveloop/motion_control_gap_audit/motorcycle_manual_feedback_motion_gap.json")
DEFAULT_ACTOR_TRACK_AUDIT = Path("outputs/driveloop/actor_track_surface_audit/tiny_real_actor_track_surface_audit.json")


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def requested_prompt_motions(prompt: str) -> list[str]:
    motions: list[str] = []
    for name, rule in RULES.items():
        if rule.get("type") != "motion":
            continue
        if contains_any(prompt, list(rule.get("prompt_aliases", []))):
            motions.append(name)
    return list(dict.fromkeys(motions))


def first_dict(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


def runtime_input_audit(summary: dict[str, Any]) -> dict[str, Any]:
    metadata = summary.get("metadata", {})
    return first_dict(
        summary if summary.get("schema_version") == "dd2_runtime_input_audit.v0" else None,
        summary.get("runtime_input_audit"),
        summary.get("dd2_runtime_input_audit"),
        metadata.get("dd2_runtime_input_audit") if isinstance(metadata, dict) else None,
    )


def paper_stage_3(summary: dict[str, Any]) -> dict[str, Any]:
    return first_dict(summary.get("paper_alignment_stage_3"))


def has_available_tensor(runtime: dict[str, Any], names: list[str]) -> bool:
    for name in names:
        value = runtime.get(name)
        if isinstance(value, dict) and value.get("available") is True:
            return True
    return False


def build_audit(
    prompt: str,
    backend_summary: dict[str, Any],
    velocity_audit: dict[str, Any] | None = None,
    motion_gap: dict[str, Any] | None = None,
    actor_track_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    velocity_audit = velocity_audit or {}
    motion_gap = motion_gap or {}
    actor_track_audit = actor_track_audit or {}

    requested_motions = requested_prompt_motions(prompt)
    runtime = runtime_input_audit(backend_summary)
    stage_3 = paper_stage_3(backend_summary)

    velocity_claim = first_dict(velocity_audit.get("claim"))
    motion_gap_claim = first_dict(motion_gap.get("claim"))
    control_path_status = first_dict(motion_gap.get("control_path_status"))
    actor_track_claim = first_dict(actor_track_audit.get("claim"))
    actor_track_surface = first_dict(actor_track_audit.get("track_surface"))

    box_condition_available = has_available_tensor(runtime, ["box_downsampler_input"])
    grounding_condition_available = has_available_tensor(runtime, ["grounding_downsampler_input"])
    trajectory_tensor_available = has_available_tensor(
        runtime,
        [
            "trajectory",
            "trajectories",
            "trajectory_tensor",
            "actor_trajectory",
            "future_trajectory",
            "displacement",
            "actor_displacement",
        ],
    )
    velocity_tensor_available = has_available_tensor(runtime, ["velocity", "velocities", "actor_velocity"])

    dataset_velocity_exists = velocity_claim.get("dataset_velocity_surface_available") is True or (
        velocity_claim.get("velocity_surface_available") is True
    )
    velocity_consumed_claimed = velocity_claim.get("velocity_consumed_by_dd2_runtime") is True
    velocity_consumed = velocity_tensor_available and velocity_consumed_claimed

    per_frame_actor_identity = actor_track_claim.get("per_frame_actor_identity_observed") is True
    per_frame_actor_boxes3d = actor_track_claim.get("per_frame_actor_boxes3d_grouped_by_identity") is True
    hdmap_override_verified = False

    blockers: list[str] = []
    if requested_motions:
        if not trajectory_tensor_available:
            blockers.append("trajectory_tensor_not_observed_in_runtime_audit")
        if not velocity_tensor_available and not velocity_consumed:
            blockers.append("velocity_or_displacement_tensor_not_consumed_by_runtime")
        if control_path_status.get("velocity_motion_control") == "observed_only_not_condition_tensor":
            blockers.append("velocity_metadata_observed_only_not_condition_tensor")
        if not per_frame_actor_identity:
            blockers.append("per_frame_actor_identity_not_observed")
        if not per_frame_actor_boxes3d:
            blockers.append("per_frame_actor_boxes3d_not_verified")
        if control_path_status.get("boxes3d_target_override") == "not_applied":
            blockers.append("target_boxes3d_override_not_applied")
        if not hdmap_override_verified:
            blockers.append("hdmap_lane_geometry_override_not_verified")
        if box_condition_available:
            blockers.append("static_box_condition_available_but_not_temporal_motion_control")
        if motion_gap_claim.get("semantic_success_claim_allowed") is False:
            blockers.append("semantic_success_claim_not_allowed_by_motion_gap")

    if not requested_motions:
        status = "not_applicable"
        status_reason = "accepted prompt does not request a known trajectory/motion primitive"
    elif blockers:
        status = "not_runtime_connected"
        status_reason = "requested motion is not backed by verified DD2 runtime trajectory surfaces"
    else:
        status = "runtime_connected"

    return {
        "schema_version": "driveloop_trajectory_runtime_surface_audit.v0",
        "accepted_prompt": prompt,
        "requested_motions": requested_motions,
        "status": status,
        "status_reason": status_reason,
        "surfaces": {
            "box_condition": {
                "available": box_condition_available,
                "interpretation": "static/spatial conditioning only; does not prove temporal motion",
            },
            "grounding_condition": {
                "available": grounding_condition_available,
                "interpretation": "conditioning tensor available; decoded trajectory semantics not proven",
            },
            "trajectory_tensor": {
                "available": trajectory_tensor_available,
                "runtime_keys_checked": [
                    "trajectory",
                    "trajectories",
                    "trajectory_tensor",
                    "actor_trajectory",
                    "future_trajectory",
                    "displacement",
                    "actor_displacement",
                ],
            },
            "velocity_tensor": {
                "available_in_runtime_audit": velocity_tensor_available,
                "dataset_velocity_surface_available": dataset_velocity_exists,
                "velocity_consumed_by_dd2_runtime": velocity_consumed,
                "velocity_consumed_claimed_by_velocity_audit": velocity_consumed_claimed,
                "velocity_runtime_consumption_requires_runtime_tensor": True,
            },
            "actor_track_identity": {
                "per_frame_actor_identity_observed": per_frame_actor_identity,
                "actor_track_audit_status": actor_track_audit.get("status", "unknown"),
                "persistent_track_count": actor_track_surface.get("persistent_track_count"),
            },
            "per_frame_actor_boxes3d": {
                "verified": per_frame_actor_boxes3d,
                "current_surface": "grouped_by_instance_token"
                if per_frame_actor_boxes3d
                else ("image_box_condition_only_not_per_frame_actor_motion" if box_condition_available else "not_observed"),
            },
            "hdmap_lane_geometry": {
                "override_verified": hdmap_override_verified,
                "motion_gap_status": control_path_status.get("image_hdmap", "unknown"),
            },
        },
        "source_signals": {
            "motion_gap_lane_change_motion_tensor_control": motion_gap_claim.get(
                "lane_change_motion_tensor_control"
            ),
            "motion_gap_semantic_lane_change_claim": motion_gap_claim.get("semantic_lane_change_claim"),
            "motion_gap_semantic_success_claim_allowed": motion_gap_claim.get("semantic_success_claim_allowed"),
            "motion_gap_boxes3d_target_override": control_path_status.get("boxes3d_target_override"),
            "motion_gap_velocity_motion_control": control_path_status.get("velocity_motion_control"),
            "paper_stage_3_status": stage_3.get("status"),
            "paper_stage_3_tensor_control_ready": stage_3.get("tensor_control_ready"),
            "actor_track_audit_status": actor_track_audit.get("status"),
        },
        "blockers": list(dict.fromkeys(blockers)),
        "claim_boundary": {
            "trajectory_surface_audit_is_not_video_semantic_success": True,
            "static_boxes_are_not_temporal_motion_control": True,
            "runtime_tensor_hashes_do_not_decode_lane_change": True,
            "semantic_success_requires_measured_passed_review": True,
        },
        "next_required_steps": next_steps(status, blockers),
    }


def next_steps(status: str, blockers: list[str]) -> list[str]:
    if status == "not_applicable":
        return ["use a prompt with explicit motion requirements before trajectory surface audit"]

    steps: list[str] = []
    if "trajectory_tensor_not_observed_in_runtime_audit" in blockers:
        steps.append("identify or add a DD2 runtime trajectory/displacement surface before claiming lane-change control")
    if "velocity_or_displacement_tensor_not_consumed_by_runtime" in blockers:
        steps.append("connect dataset velocity/displacement evidence to DD2 runtime or record it as unavailable")
    if "per_frame_actor_identity_not_observed" in blockers:
        steps.append("audit actor track identity across frames")
    if "per_frame_actor_boxes3d_not_verified" in blockers:
        steps.append("audit per-frame actor boxes3d instead of static sample-level boxes only")
    if "target_boxes3d_override_not_applied" in blockers:
        steps.append("do not treat baseline runtime boxes3d as target actor override")
    if "velocity_metadata_observed_only_not_condition_tensor" in blockers:
        steps.append("keep dataset velocity as metadata until it is connected to a runtime condition tensor")
    if "semantic_success_claim_not_allowed_by_motion_gap" in blockers:
        steps.append("keep semantic_success_claim_allowed false until measured review passes")
    if "hdmap_lane_geometry_override_not_verified" in blockers:
        steps.append("audit HDMap/lane geometry compatibility with requested motion")
    if "static_box_condition_available_but_not_temporal_motion_control" in blockers:
        steps.append("do not use static boxes3d changes as lane-change motion evidence")

    steps.append("run audit-only/runtime tensor checks before any new GPU candidate")
    steps.append("record measured_failed if generated video still lacks requested motion")
    return list(dict.fromkeys(steps))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether requested prompt motion is connected to DD2 runtime trajectory surfaces.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--backend-summary", type=Path, default=DEFAULT_BACKEND_SUMMARY)
    parser.add_argument("--velocity-audit", type=Path, default=DEFAULT_VELOCITY_AUDIT)
    parser.add_argument("--motion-gap", type=Path, default=DEFAULT_MOTION_GAP)
    parser.add_argument("--actor-track-audit", type=Path, default=DEFAULT_ACTOR_TRACK_AUDIT)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    audit = build_audit(
        prompt=args.prompt,
        backend_summary=load_json(args.backend_summary),
        velocity_audit=load_json(args.velocity_audit),
        motion_gap=load_json(args.motion_gap),
        actor_track_audit=load_json(args.actor_track_audit),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(args.output)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
