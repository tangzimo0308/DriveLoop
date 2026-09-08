from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from driveloop.dd2_override import read_override_audit
from scripts.run_candidate70_hdmap_geometry_introspection_audit import load_json
from scripts.run_dd2_structural_audit import load_config, tensor_sig, transform_first_sample


DEFAULT_CONFIG = Path("dreamer-train/projects/DriveDreamer2/configs/drivedreamer2_img_cond_mini_local.py")
DEFAULT_CANDIDATE = Path(
    "outputs/driveloop/candidate70_hdmap_lane_geometry_replacement_candidate/"
    "candidate70_lane_geometry_replacement_candidate_summary.json"
)
DEFAULT_OUTPUT = Path(
    "outputs/driveloop/candidate70_hdmap_lane_geometry_replacement_surface_audit/"
    "candidate70_lane_geometry_replacement_candidate_to_grounding_surface.json"
)


def select_candidate_record(candidate_summary: dict[str, Any], frame_index: int) -> dict[str, Any]:
    records = candidate_summary.get("records", [])
    if not isinstance(records, list) or not records:
        return {"available": False, "reason": "missing_candidate_records", "frame_index": frame_index}
    if frame_index < 0 or frame_index >= len(records):
        return {"available": False, "reason": "frame_index_out_of_range", "frame_index": frame_index}

    record = records[frame_index] if isinstance(records[frame_index], dict) else {}
    candidate_signature = record.get("candidate_signature", {})
    operation = record.get("operation", {})
    raster_path = record.get("candidate_raster_path")
    raster_exists = Path(str(raster_path)).exists() if raster_path else False

    available = (
        candidate_summary.get("claim", {}).get("candidate70_geometry_grounded_replacement_candidate_available") is True
        and operation.get("operation") == "offset_lane_divider_local_map_vector_before_camera_projection"
        and operation.get("coordinate_frame") == "ego_aligned_local_map_patch"
        and record.get("baseline_matches_converter_signature") is True
        and record.get("candidate_differs_from_baseline") is True
        and int(record.get("diff_nonzero") or 0) > 0
        and int(operation.get("modified_visible_count") or 0) > 0
        and raster_path
        and raster_exists
        and candidate_signature.get("sha256")
    )

    return {
        "available": bool(available),
        "reason": "available" if available else "unverified_candidate_record",
        "frame_index": frame_index,
        "data_index": record.get("data_index"),
        "frame_idx": record.get("frame_idx"),
        "path": raster_path,
        "path_exists": raster_exists,
        "expected_sha256": candidate_signature.get("sha256"),
        "candidate_signature": candidate_signature,
        "baseline_matches_converter_signature": record.get("baseline_matches_converter_signature"),
        "candidate_differs_from_baseline": record.get("candidate_differs_from_baseline"),
        "diff_nonzero": record.get("diff_nonzero"),
        "operation": operation,
        "source": "candidate70_lane_geometry_replacement_candidate.candidate_raster_path",
        "provenance": "ego_aligned_local_map_vector_offset_before_camera_projection",
        "claim_boundary": {
            "candidate_is_local_map_vector_geometry_operation": True,
            "candidate_is_not_direct_nuscenes_database_edit": True,
            "candidate_surface_audit_is_not_video_semantic_success": True,
        },
    }


def build_override_json(candidate_record: dict[str, Any]) -> dict[str, Any]:
    if candidate_record.get("available") is not True:
        raise ValueError(f"Refusing unavailable candidate raster: {candidate_record.get('reason')}")

    return {
        "available": True,
        "schema_version": "driveloop_dd2_override.v0",
        "source": "candidate70_hdmap_lane_geometry_replacement_surface_audit",
        "image_hdmap": {
            "mode": "replace_from_path",
            "path": candidate_record["path"],
            "source": candidate_record["source"],
            "provenance": candidate_record["provenance"],
            "expected_sha256": candidate_record["expected_sha256"],
        },
        "boxes3d": {
            "mode": "append",
            "append": [],
            "source": "not_modified",
        },
        "audit": {
            "control_level": "runtime_surface_replacement_audit",
            "limitations": [
                "local_map_vector_geometry_replacement_is_not_lane_change_control",
                "grounding_surface_hash_change_does_not_prove_lane_change_control",
                "runtime_tensor_audit_is_not_video_semantic_success",
            ],
        },
    }


def signature_changed(baseline: dict[str, Any], replacement: dict[str, Any], key: str) -> bool:
    return baseline.get(key, {}).get("sha256") != replacement.get(key, {}).get("sha256")


def first_hdmap_override_entry(override_audit: dict[str, Any]) -> dict[str, Any]:
    for entry in override_audit.get("entries_preview", []):
        if not isinstance(entry, dict):
            continue
        for key in ("applied", "skipped"):
            for item in entry.get(key, []):
                if isinstance(item, dict) and item.get("target") == "image_hdmap":
                    return item
    return {}


def build_report(
    baseline_signatures: dict[str, Any],
    replacement_signatures: dict[str, Any],
    override_audit: dict[str, Any],
    candidate_summary: dict[str, Any],
    candidate_record: dict[str, Any],
) -> dict[str, Any]:
    grounding_changed = signature_changed(baseline_signatures, replacement_signatures, "grounding_downsampler_input")
    box_changed = signature_changed(baseline_signatures, replacement_signatures, "box_downsampler_input")
    image_changed = signature_changed(baseline_signatures, replacement_signatures, "input_image")
    changed_counts = override_audit.get("changed_counts", {})
    image_hdmap_changed = bool(changed_counts.get("image_hdmap"))
    reaches_grounding = candidate_record.get("available") is True and image_hdmap_changed and grounding_changed

    return {
        "schema_version": "candidate70_hdmap_lane_geometry_replacement_surface_audit.v1",
        "status": "local_map_vector_lane_geometry_replacement_reaches_grounding_surface" if reaches_grounding else "not_observed",
        "audit_only": True,
        "does_not_run_gpu": True,
        "does_not_generate_video": True,
        "candidate_source": candidate_record,
        "intervention": {
            "target": "image_hdmap",
            "mode": "replace_from_path",
            "path": candidate_record.get("path"),
            "source": candidate_record.get("source"),
            "provenance": candidate_record.get("provenance"),
            "expected_sha256": candidate_record.get("expected_sha256"),
            "override_audit_entry": first_hdmap_override_entry(override_audit),
        },
        "surfaces": {
            "image_hdmap_override": {
                "changed": image_hdmap_changed,
                "override_audit_available": override_audit.get("available") is True,
                "changed_counts": changed_counts,
            },
            "grounding_downsampler_input": {
                "changed": grounding_changed,
                "baseline": baseline_signatures.get("grounding_downsampler_input"),
                "replacement": replacement_signatures.get("grounding_downsampler_input"),
            },
            "box_downsampler_input": {
                "changed": box_changed,
                "expected_changed": False,
            },
            "input_image": {
                "changed": image_changed,
                "expected_changed": False,
            },
        },
        "claim": {
            "candidate70_geometry_grounded_replacement_candidate_available": (
                candidate_summary.get("claim", {}).get("candidate70_geometry_grounded_replacement_candidate_available") is True
            ),
            "candidate70_local_map_vector_lane_geometry_replacement_reaches_grounding_downsampler_input": reaches_grounding,
            "candidate70_true_lane_geometry_replacement_available": reaches_grounding,
            "hdmap_lane_geometry_override_verified": reaches_grounding,
            "lane_change_control_verified": False,
            "runtime_motion_control_connected": False,
            "semantic_success_claim_allowed": False,
        },
        "claim_boundary": {
            "local_map_vector_lane_geometry_replacement_is_not_gpu_approval": True,
            "local_map_vector_lane_geometry_replacement_is_not_lane_change_control": True,
            "grounding_surface_hash_change_is_not_lane_change_control": True,
            "runtime_tensor_audit_is_not_video_semantic_success": True,
            "semantic_success_requires_measured_passed_review": True,
        },
        "next_required_steps": [
            "Record that the local-map-vector lane geometry replacement reaches the DD2 grounding surface if observed.",
            "Gate may use this only as HDMap tensor-surface readiness evidence, not semantic success evidence.",
            "Do not claim video semantic success without measured evaluation.",
            "Do not run GPU without explicit user approval.",
        ],
    }


def run_audit(config_path: Path, candidate_path: Path, frame_index: int, output_path: Path) -> dict[str, Any]:
    candidate_summary = load_json(candidate_path)
    candidate_record = select_candidate_record(candidate_summary, frame_index)
    override_json = build_override_json(candidate_record)

    config = load_config(config_path)
    data_cfg = config["dataloaders"]["test"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    override_audit_path = output_path.with_suffix(".override_audit.jsonl")

    try:
        baseline = transform_first_sample(data_cfg, override_json=None, audit_path=None)
        replacement = transform_first_sample(data_cfg, override_json=override_json, audit_path=override_audit_path)

        keys = ["grounding_downsampler_input", "box_downsampler_input", "input_image"]
        baseline_signatures = {key: tensor_sig(baseline[key]) for key in keys}
        replacement_signatures = {key: tensor_sig(replacement[key]) for key in keys}
        override_audit = read_override_audit(override_audit_path)

        report = build_report(
            baseline_signatures=baseline_signatures,
            replacement_signatures=replacement_signatures,
            override_audit=override_audit,
            candidate_summary=candidate_summary,
            candidate_record=candidate_record,
        )
        report["inputs"] = {
            "config_path": str(config_path),
            "candidate_path": str(candidate_path),
            "frame_index": frame_index,
            "override_audit_path": str(override_audit_path),
        }
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        os.environ.pop("DRIVELOOP_DD2_OVERRIDE_JSON", None)
        os.environ.pop("DRIVELOOP_DD2_OVERRIDE_AUDIT_PATH", None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether a local-map-vector candidate70 HDMap raster reaches DD2 grounding_downsampler_input.")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--candidate-path", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = run_audit(
        config_path=args.config_path,
        candidate_path=args.candidate_path,
        frame_index=args.frame_index,
        output_path=args.output,
    )
    print(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
