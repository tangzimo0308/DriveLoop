from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from driveloop.dd2_override import read_override_audit
from scripts.run_dd2_structural_audit import load_config, tensor_sig, transform_first_sample


DEFAULT_CONFIG = "dreamer-train/projects/DriveDreamer2/configs/drivedreamer2_img_cond_mini_local.py"
DEFAULT_PROBE = "outputs/driveloop/candidate70_hdmap_raster_probe/candidate70_hdmap_raster_probe_summary.json"
DEFAULT_OUTPUT = "outputs/driveloop/hdmap_replacement_surface_audit/candidate70_verified_raster_to_grounding_surface.json"


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def signature_changed(baseline: dict[str, Any], replacement: dict[str, Any], key: str) -> bool:
    return baseline.get(key, {}).get("sha256") != replacement.get(key, {}).get("sha256")


def select_verified_raster_record(probe: dict[str, Any], frame_index: int) -> dict[str, Any]:
    records = probe.get("records", [])
    if not isinstance(records, list) or not records:
        return {"verified": False, "reason": "missing_probe_records", "frame_index": frame_index}
    if frame_index < 0 or frame_index >= len(records):
        return {"verified": False, "reason": "frame_index_out_of_range", "frame_index": frame_index}

    record = records[frame_index] if isinstance(records[frame_index], dict) else {}
    converter_signature = record.get("converter_signature", {})
    if not isinstance(converter_signature, dict):
        converter_signature = {}
    processed_matches = record.get("processed_matches", [])
    if not isinstance(processed_matches, list):
        processed_matches = []

    raster_path = record.get("converter_hdmap_path")
    expected_sha256 = converter_signature.get("sha256")
    nonzero = converter_signature.get("nonzero") or 0
    processed_match_true = sum(1 for item in processed_matches if isinstance(item, dict) and item.get("matches_converter") is True)
    processed_match_false = sum(1 for item in processed_matches if isinstance(item, dict) and item.get("matches_converter") is False)
    processed_matches_converter = bool(processed_matches) and processed_match_false == 0 and processed_match_true == len(processed_matches)
    raster_exists = Path(str(raster_path)).exists() if raster_path else False

    verified = bool(raster_path and raster_exists and expected_sha256 and nonzero > 0 and processed_matches_converter)
    reason = "verified" if verified else "unverified_raster_source"

    return {
        "verified": verified,
        "reason": reason,
        "frame_index": frame_index,
        "data_index": record.get("data_index"),
        "frame_idx": record.get("frame_idx"),
        "path": raster_path,
        "path_exists": raster_exists,
        "expected_sha256": expected_sha256,
        "converter_signature": converter_signature,
        "processed_match_true": processed_match_true,
        "processed_match_false": processed_match_false,
        "processed_matches_converter": processed_matches_converter,
        "source": "candidate70_hdmap_raster_probe.converter_hdmap_path",
        "provenance": "converter_generated_raster_matches_processed_hdmap_lmdb_by_sha256",
    }


def build_override_json(verified_raster: dict[str, Any]) -> dict[str, Any]:
    if verified_raster.get("verified") is not True:
        raise ValueError(f"Refusing unverified raster source: {verified_raster.get('reason')}")

    return {
        "available": True,
        "schema_version": "driveloop_dd2_override.v0",
        "source": "hdmap_replacement_surface_audit",
        "image_hdmap": {
            "mode": "replace_from_path",
            "path": verified_raster["path"],
            "source": verified_raster["source"],
            "provenance": verified_raster["provenance"],
            "expected_sha256": verified_raster["expected_sha256"],
        },
        "boxes3d": {
            "mode": "append",
            "append": [],
            "source": "not_modified",
        },
        "audit": {
            "control_level": "runtime_surface_replacement_audit",
            "limitations": [
                "verified_raster_replacement_is_not_lane_geometry_override",
                "grounding_surface_hash_change_does_not_prove_lane_change_control",
                "runtime_tensor_audit_is_not_video_semantic_success",
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
    verified_raster: dict[str, Any],
) -> dict[str, Any]:
    grounding_changed = signature_changed(
        baseline_signatures,
        replacement_signatures,
        "grounding_downsampler_input",
    )
    box_changed = signature_changed(
        baseline_signatures,
        replacement_signatures,
        "box_downsampler_input",
    )
    image_changed = signature_changed(
        baseline_signatures,
        replacement_signatures,
        "input_image",
    )
    changed_counts = override_audit.get("changed_counts", {})
    hdmap_override_changed = bool(changed_counts.get("image_hdmap"))
    hdmap_override_entry = first_hdmap_override_entry(override_audit)

    status = (
        "replacement_raster_reaches_grounding_surface"
        if verified_raster.get("verified") is True and hdmap_override_changed and grounding_changed
        else "not_observed"
    )

    return {
        "schema_version": "driveloop_hdmap_replacement_surface_audit.v0",
        "status": status,
        "does_not_run_gpu": True,
        "does_not_generate_video": True,
        "verified_raster_source": verified_raster,
        "intervention": {
            "target": "image_hdmap",
            "mode": "replace_from_path",
            "path": verified_raster.get("path"),
            "source": verified_raster.get("source"),
            "provenance": verified_raster.get("provenance"),
            "expected_sha256": verified_raster.get("expected_sha256"),
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
                "interpretation": "Replacement raster reaches the DD2 grounding downsampler runtime surface if changed is true.",
            },
            "box_downsampler_input": {
                "changed": box_changed,
                "interpretation": "Expected to remain unchanged for an HDMap-only replacement audit.",
            },
            "input_image": {
                "changed": image_changed,
                "interpretation": "Expected to remain unchanged for a structural-condition audit.",
            },
        },
        "claim": {
            "replacement_raster_reaches_grounding_downsampler_input": status == "replacement_raster_reaches_grounding_surface",
            "candidate70_verified_replacement_hdmap_raster_available": False,
            "hdmap_lane_geometry_override_verified": False,
            "lane_change_control_verified": False,
            "runtime_motion_control_connected": False,
            "video_semantic_claim": "not_evaluated_by_this_audit",
            "semantic_success_claim_allowed": False,
        },
        "claim_boundary": {
            "verified_raster_replacement_is_not_lane_geometry_override": True,
            "grounding_surface_hash_change_is_not_lane_change_control": True,
            "runtime_tensor_audit_is_not_video_semantic_success": True,
            "semantic_success_requires_measured_passed_review": True,
        },
        "next_required_steps": [
            "Record only that a verified raster can be loaded and observed at grounding_downsampler_input.",
            "Do not claim HDMap lane geometry override from this audit.",
            "Construct a true replacement lane-geometry raster separately before any lane-change intervention claim.",
            "Do not run GPU until the replacement source and audit boundary are recorded.",
        ],
    }


def run_audit(config_path: Path, probe_path: Path, frame_index: int, output_path: Path) -> dict[str, Any]:
    probe = load_json(probe_path)
    verified_raster = select_verified_raster_record(probe, frame_index)
    override_json = build_override_json(verified_raster)

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
            verified_raster=verified_raster,
        )
        report["inputs"] = {
            "config_path": str(config_path),
            "probe_path": str(probe_path),
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
        description="Audit whether a verified replacement image_hdmap raster reaches DD2 grounding_downsampler_input."
    )
    parser.add_argument("--config-path", type=Path, default=Path(DEFAULT_CONFIG))
    parser.add_argument("--probe-path", type=Path, default=Path(DEFAULT_PROBE))
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    args = parser.parse_args()

    report = run_audit(args.config_path, args.probe_path, args.frame_index, args.output)
    print(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
