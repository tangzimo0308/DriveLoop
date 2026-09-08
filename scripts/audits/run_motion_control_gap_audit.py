from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def first_dict(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


def first_list(*values: Any) -> list[Any]:
    for value in values:
        if isinstance(value, list):
            return value
    return []


def nested(data: dict[str, Any], *keys: str) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def infer_paper_report_path(summary_path: Path) -> Path | None:
    candidate = summary_path.parent / "paper_alignment_report_00.json"
    return candidate if candidate.exists() else None


def infer_override_audit_path(summary_path: Path) -> Path | None:
    if summary_path.name.startswith("dd2_runtime_input_audit_") and summary_path.suffix == ".json":
        index = summary_path.stem[len("dd2_runtime_input_audit_") :]
        candidate = summary_path.with_name(f"dd2_override_audit_{index}.jsonl")
        if candidate.exists():
            return candidate
    candidate = summary_path.parent / "dd2_override_audit_00.jsonl"
    return candidate if candidate.exists() else None


def load_override_audit(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}

    entries = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    changed_counts: dict[str, int] = {}
    for entry in entries:
        changed_map = first_dict(entry.get("changed"))
        for target, changed in changed_map.items():
            if changed:
                changed_counts[target] = changed_counts.get(target, 0) + 1
        if "image_box" not in changed_map and entry.get("image_box_expected_changed"):
            changed_counts["image_box"] = changed_counts.get("image_box", 0) + 1

    return {
        "available": True,
        "path": str(path),
        "entry_count": len(entries),
        "changed_counts": changed_counts,
        "entries_preview": entries[:3],
    }


def runtime_hashes(runtime: dict[str, Any]) -> dict[str, Any]:
    return {
        name: nested(runtime, name, "sha256")
        for name in [
            "prompt_embed",
            "box_downsampler_input",
            "grounding_downsampler_input",
            "img_cond",
        ]
    }


def _runtime_tensor_connected(runtime: dict[str, Any], name: str) -> bool:
    return bool(nested(runtime, name, "sha256")) or bool(nested(runtime, name, "available"))


def _motion_metadata_status(motion_metadata: dict[str, Any]) -> dict[str, str]:
    velocities_observed = bool(motion_metadata.get("velocities_available_in_batch_any"))
    actor_identity_observed = bool(motion_metadata.get("actor_identity_available_in_batch_any"))
    per_frame_boxes_observed = bool(motion_metadata.get("per_frame_actor_boxes3d_observed_any"))

    return {
        "velocity_motion_control": (
            "observed_only_not_condition_tensor" if velocities_observed else "not_observed"
        ),
        "actor_identity": (
            "observed_only_not_runtime_control" if actor_identity_observed else "not_observed"
        ),
        "per_frame_actor_boxes3d": (
            "observed_only_not_trajectory_control" if per_frame_boxes_observed else "not_observed"
        ),
    }


def build_motion_control_gap_report(
    summary_path: Path,
    paper_alignment_report_path: Path | None = None,
) -> dict[str, Any]:
    data = load_json(summary_path)
    metadata = first_dict(data.get("metadata"))

    paper_path = paper_alignment_report_path or infer_paper_report_path(summary_path)
    paper = load_json(paper_path) if paper_path and paper_path.exists() else {}
    stage3 = first_dict(paper.get("stage_3_scene_consistent_generation"))

    exec_cond = first_dict(
        nested(data, "metadata", "dd2_executable_condition"),
        data.get("dd2_executable_condition"),
        data.get("executable_condition"),
    )
    trace = first_dict(exec_cond.get("trace_metadata"))
    structural_plan = first_dict(
        exec_cond.get("structural_input_plan"),
        data.get("dd2_structural_input_plan"),
        metadata.get("dd2_structural_input_plan"),
        stage3.get("structural_input_plan"),
    )
    runtime = first_dict(
        data if data.get("schema_version") == "dd2_runtime_input_audit.v0" else None,
        data.get("runtime_input_audit"),
        nested(data, "metadata", "dd2_runtime_input_audit"),
    )
    override_audit_path = infer_override_audit_path(summary_path)
    override = first_dict(
        data.get("override_audit"),
        nested(data, "metadata", "dd2_override_audit"),
        load_override_audit(override_audit_path),
    )

    changed_counts = first_dict(override.get("changed_counts"))
    motion_controls = first_list(
        exec_cond.get("motion_controls"),
        nested(data, "audit_summary", "motion_controls"),
    )

    image_hdmap = first_dict(structural_plan.get("image_hdmap"))
    image_box = first_dict(structural_plan.get("image_box"))
    motion_metadata = first_dict(runtime.get("motion_metadata"))
    motion_metadata_status = _motion_metadata_status(motion_metadata)
    boxes3d_override_applied = bool(changed_counts.get("boxes3d"))

    control_path_status = {
        "text_prompt": "connected" if runtime.get("prompt_override") else "not_observed",
        "scene_description": "connected" if changed_counts.get("scene_description") else "not_observed",
        "image_box_condition": (
            "connected" if _runtime_tensor_connected(runtime, "box_downsampler_input") else "not_observed"
        ),
        "image_box": image_box.get("source", "unknown"),
        "image_hdmap": image_hdmap.get("source", "unknown"),
        "boxes3d_target_override": "applied" if boxes3d_override_applied else "not_applied",
        "boxes3d_static_actor": "applied" if boxes3d_override_applied else "not_applied",
        "trajectory_tensor": "not_implemented",
        "temporal_actor_motion": "not_implemented",
        "semantic_lane_change_claim": "not_allowed",
    }
    control_path_status.update(motion_metadata_status)

    return {
        "schema_version": "driveloop_motion_control_gap_audit.v0",
        "source_summary": str(summary_path),
        "source_paper_alignment_report": str(paper_path) if paper_path else None,
        "scenario_id": data.get("scenario_id") or metadata.get("scenario_id"),
        "prompt": data.get("prompt") or metadata.get("prompt") or runtime.get("prompt_override"),
        "claim": {
            "lane_change_motion_tensor_control": "not_verified",
            "semantic_lane_change_claim": "not_allowed",
            "semantic_success_claim_allowed": False,
            "video_semantic_claim": "not_evaluated_by_this_audit",
            "tensor_audit_scope": "runtime_inputs_only",
        },
        "observed_signals": {
            "motion_controls": motion_controls,
            "alignment_feedback": trace.get("alignment_feedback"),
            "prompt_override": runtime.get("prompt_override"),
            "runtime_hashes": runtime_hashes(runtime),
            "runtime_motion_metadata": motion_metadata,
            "override_audit_path": str(override_audit_path) if override_audit_path else override.get("path"),
            "override_changed_counts": changed_counts,
        },
        "control_path_status": control_path_status,
        "limitations": [
            "box_downsampler_input is a DD2 structural condition, not proof of target trajectory control",
            "velocities are observed as dataset metadata and are not connected to the DD2 condition tensor",
            "target boxes3d override is not applied unless the override audit shows boxes3d changed",
            "trajectory tensor control is not implemented",
            "runtime tensor hash changes do not prove video semantic alignment",
            "manual or perception review is required for any generated video semantic claim",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--paper-alignment-report", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    report = build_motion_control_gap_report(
        summary_path=args.summary,
        paper_alignment_report_path=args.paper_alignment_report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
