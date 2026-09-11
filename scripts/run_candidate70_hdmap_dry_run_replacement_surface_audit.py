from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from driveloop.dd2_override import read_override_audit
from scripts.run_dd2_structural_audit import load_config, tensor_sig, transform_first_sample


DEFAULT_CONFIG = Path("dreamer-train/projects/DriveDreamer2/configs/drivedreamer2_img_cond_mini_local.py")
DEFAULT_DRY_RUN = Path("outputs/driveloop/candidate70_hdmap_lane_divider_dry_run/candidate70_lane_divider_dry_run_summary.json")
DEFAULT_OUTPUT = Path("outputs/driveloop/candidate70_hdmap_dry_run_replacement_surface_audit/candidate70_dry_run_raster_to_grounding_surface.json")


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def signature_changed(baseline: dict[str, Any], replacement: dict[str, Any], key: str) -> bool:
    return baseline.get(key, {}).get("sha256") != replacement.get(key, {}).get("sha256")


def select_dry_run_candidate_record(dry_run: dict[str, Any], frame_index: int) -> dict[str, Any]:
    records = dry_run.get("records", [])
    if not isinstance(records, list) or not records:
        return {"available": False, "reason": "missing_dry_run_records", "frame_index": frame_index}
    if frame_index < 0 or frame_index >= len(records):
        return {"available": False, "reason": "frame_index_out_of_range", "frame_index": frame_index}

    record = records[frame_index] if isinstance(records[frame_index], dict) else {}
    candidate_path = record.get("candidate_raster_path")
    candidate_signature = record.get("candidate_signature", {})
    if not isinstance(candidate_signature, dict):
        candidate_signature = {}

    baseline_matches_converter = record.get("baseline_matches_converter_signature") is True
    candidate_differs_from_baseline = record.get("candidate_differs_from_baseline") is True
    path_exists = Path(str(candidate_path)).exists() if candidate_path else False
    expected_sha256 = candidate_signature.get("sha256")
    diff_nonzero = int(record.get("diff_nonzero", 0) or 0)

    available = bool(
        candidate_path
        and path_exists
        and expected_sha256
        and baseline_matches_converter
        and candidate_differs_from_baseline
        and diff_nonzero > 0
    )

    return {
        "available": available,
        "reason": "available" if available else "dry_run_candidate_not_eligible",
        "frame_index": frame_index,
        "data_index": record.get("data_index"),
        "frame_idx": record.get("frame_idx"),
        "path": candidate_path,
        "path_exists": path_exists,
        "expected_sha256": expected_sha256,
        "candidate_signature": candidate_signature,
        "baseline_matches_converter_signature": baseline_matches_converter,
        "candidate_differs_from_baseline": candidate_differs_from_baseline,
        "diff_nonzero": diff_nonzero,
        "operation": record.get("operation"),
        "source": "candidate70_lane_divider_dry_run.candidate_raster_path",
        "provenance": "synthetic_projected_lane_divider_pixel_translation_dry_run",
        "claim_boundary": {
            "dry_run_candidate_is_synthetic_not_verified_map_geometry": True,
            "gpu_requires_separate_readiness_gate": True,
        },
    }


def build_override_json(candidate: dict[str, Any]) -> dict[str, Any]:
    if candidate.get("available") is not True:
        raise ValueError(f"Refusing unavailable dry-run candidate: {candidate.get('reason')}")

    return {
        "available": True,
        "schema_version": "driveloop_dd2_override.v0",
        "source": "candidate70_hdmap_dry_run_replacement_surface_audit",
        "image_hdmap": {
            "mode": "replace_from_path",
            "path": candidate["path"],
            "source": candidate["source"],
            "provenance": candidate["provenance"],
            "expected_sha256": candidate["expected_sha256"],
        },
        "boxes3d": {
            "mode": "append",
            "append": [],
            "source": "not_modified",
        },
        "audit": {
            "control_level": "runtime_surface_replacement_audit",
            "limitations": [
                "dry_run_candidate_is_synthetic_not_verified_map_geometry",
                "grounding_surface_hash_change_does_not_prove_lane_change_control",
                "runtime_tensor_audit_is_not_video_semantic_success",
                "gpu_requires_separate_readiness_gate",
            ],
        },
    }


def first_hdmap_override_entry(override_audit: dict[str, Any]) -> dict[str, Any]:
    for entry in override_audit.get("entries_preview", []):
        if not isinstance(entry, dict):
            continue
        for key in ("applied", "skipped"):
            items = entry.get(key, [])
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict) and item.get("target") == "image_hdmap":
                    return item
    return {}


def build_report(
    baseline_signatures: dict[str, Any],
    replacement_signatures: dict[str, Any],
    override_audit: dict[str, Any],
    dry_run_candidate: dict[str, Any],
) -> dict[str, Any]:
    grounding_changed = signature_changed(baseline_signatures, replacement_signatures, "grounding_downsampler_input")
    box_changed = signature_changed(baseline_signatures, replacement_signatures, "box_downsampler_input")
    image_changed = signature_changed(baseline_signatures, replacement_signatures, "input_image")
    changed_counts = override_audit.get("changed_counts", {})
    hdmap_override_changed = bool(changed_counts.get("image_hdmap"))
    hdmap_override_entry = first_hdmap_override_entry(override_audit)

    status = (
        "dry_run_raster_reaches_grounding_surface"
        if dry_run_candidate.get("available") is True and hdmap_override_changed and grounding_changed
        else "not_observed"
    )

    return {
        "schema_version": "candidate70_hdmap_dry_run_replacement_surface_audit.v0",
        "status": status,
        "audit_only": True,
        "does_not_run_gpu": True,
        "does_not_generate_video": True,
        "does_not_modify_model_inputs": False,
        "dry_run_candidate_source": dry_run_candidate,
        "intervention": {
            "target": "image_hdmap",
            "mode": "replace_from_path",
            "path": dry_run_candidate.get("path"),
            "source": dry_run_candidate.get("source"),
            "provenance": dry_run_candidate.get("provenance"),
            "expected_sha256": dry_run_candidate.get("expected_sha256"),
            "override_audit_entry": hdmap_override_entry,
        },
        "surfaces": {
            "image_hdmap_override": {
                "changed": hdmap_override_changed,
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
            },
            "input_image": {
                "changed": image_changed,
            },
        },
        "claim": {
            "candidate70_dry_run_raster_reaches_grounding_downsampler_input": status == "dry_run_raster_reaches_grounding_surface",
            "candidate70_true_lane_geometry_replacement_available": False,
            "hdmap_lane_geometry_override_verified": False,
            "lane_change_control_verified": False,
            "runtime_motion_control_connected": False,
            "semantic_success_claim_allowed": False,
        },
        "claim_boundary": {
            "dry_run_candidate_is_synthetic_not_verified_map_geometry": True,
            "grounding_surface_hash_change_is_not_lane_change_control": True,
            "runtime_tensor_audit_is_not_video_semantic_success": True,
            "gpu_requires_separate_readiness_gate": True,
        },
        "next_required_steps": [
            "Record only that the dry-run raster can be loaded and observed at grounding_downsampler_input.",
            "Do not claim HDMap lane geometry override from this audit.",
            "Run GPU only after true replacement evidence, readiness gate, and explicit user approval.",
        ],
    }


def run_audit(config_path: Path, dry_run_path: Path, frame_index: int, output_path: Path) -> dict[str, Any]:
    dry_run = load_json(dry_run_path)
    candidate = select_dry_run_candidate_record(dry_run, frame_index)
    override_json = build_override_json(candidate)

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
            dry_run_candidate=candidate,
        )
        report["inputs"] = {
            "config_path": str(config_path),
            "dry_run_path": str(dry_run_path),
            "frame_index": frame_index,
            "override_audit_path": str(override_audit_path),
        }
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        os.environ.pop("DRIVELOOP_DD2_OVERRIDE_JSON", None)
        os.environ.pop("DRIVELOOP_DD2_OVERRIDE_AUDIT_PATH", None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether a candidate70 dry-run HDMap raster reaches DD2 grounding_downsampler_input."
    )
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run-path", type=Path, default=DEFAULT_DRY_RUN)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = run_audit(args.config_path, args.dry_run_path, args.frame_index, args.output)
    print(args.output)
    print(json.dumps({
        "schema_version": report["schema_version"],
        "status": report["status"],
        "does_not_run_gpu": report["does_not_run_gpu"],
        "dry_run_candidate_available": report["dry_run_candidate_source"]["available"],
        "grounding_downsampler_input_changed": report["surfaces"]["grounding_downsampler_input"]["changed"],
        "box_downsampler_input_changed": report["surfaces"]["box_downsampler_input"]["changed"],
        "input_image_changed": report["surfaces"]["input_image"]["changed"],
        "claim": report["claim"],
        "claim_boundary": report["claim_boundary"],
    }, indent=2))


if __name__ == "__main__":
    main()
