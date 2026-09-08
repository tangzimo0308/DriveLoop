from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dd_scripts.converters.nuscenes_converter import NuScenesConverter
from scripts.run_candidate70_hdmap_geometry_introspection_audit import (
    DEFAULT_PROBE,
    array_signature,
    build_filtered_vectors,
    load_json,
    rasterize_vectors,
)


DEFAULT_OUTPUT = Path("outputs/driveloop/candidate70_hdmap_lane_divider_dry_run/candidate70_lane_divider_dry_run_summary.json")
DEFAULT_IMAGES_DIR = Path("outputs/driveloop/candidate70_hdmap_lane_divider_dry_run/images")


def perturb_lane_dividers(
    vectors: list[dict[str, Any]],
    dx_pixels: float,
    dy_pixels: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_vectors = copy.deepcopy(vectors)
    modified_count = 0
    modified_visible_count = 0

    for vector in candidate_vectors:
        if vector.get("type_name") != "lane_divider":
            continue
        pts = np.asarray(vector.get("pts", []), dtype=np.float64)
        pts_num = int(vector.get("pts_num", 0) or 0)
        if pts.size == 0 or pts_num < 2:
            continue
        vector["pts"] = pts + np.array([dx_pixels, dy_pixels], dtype=np.float64)
        vector["dry_run_operation"] = {
            "operation": "translate_projected_lane_divider_pixels",
            "dx_pixels": dx_pixels,
            "dy_pixels": dy_pixels,
        }
        modified_count += 1
        modified_visible_count += 1

    return candidate_vectors, {
        "operation": "translate_projected_lane_divider_pixels",
        "target_type_name": "lane_divider",
        "dx_pixels": dx_pixels,
        "dy_pixels": dy_pixels,
        "modified_count": modified_count,
        "modified_visible_count": modified_visible_count,
    }


def diff_image(a: Image.Image, b: Image.Image) -> Image.Image:
    arr_a = np.asarray(a.convert("RGB"), dtype=np.int16)
    arr_b = np.asarray(b.convert("RGB"), dtype=np.int16)
    diff = np.abs(arr_a - arr_b).astype(np.uint8)
    return Image.fromarray(diff)


def build_frame_record(
    converter: NuScenesConverter,
    probe_record: dict[str, Any],
    images_dir: Path,
    dx_pixels: float,
    dy_pixels: float,
) -> dict[str, Any]:
    cam_token = str(probe_record["cam_token"])
    scene_token = str(probe_record["scene_token"])
    vectors, metadata = build_filtered_vectors(converter, cam_token, scene_token)

    baseline = rasterize_vectors(vectors, metadata["imsize"])
    candidate_vectors, operation = perturb_lane_dividers(vectors, dx_pixels=dx_pixels, dy_pixels=dy_pixels)
    candidate = rasterize_vectors(candidate_vectors, metadata["imsize"])
    diff = diff_image(baseline, candidate)

    frame_index = int(probe_record.get("i", 0))
    data_index = probe_record.get("data_index")
    frame_prefix = f"candidate70_frame_{frame_index:02d}_idx_{data_index}"
    images_dir.mkdir(parents=True, exist_ok=True)

    candidate_path = images_dir / f"{frame_prefix}_lane_divider_dry_run_candidate.png"
    diff_path = images_dir / f"{frame_prefix}_lane_divider_dry_run_diff.png"
    candidate.save(candidate_path)
    diff.save(diff_path)

    baseline_signature = array_signature(baseline)
    candidate_signature = array_signature(candidate)
    diff_signature = array_signature(diff)
    converter_signature = probe_record.get("converter_signature", {})
    baseline_matches_converter = baseline_signature.get("sha256") == converter_signature.get("sha256")
    candidate_differs_from_baseline = candidate_signature.get("sha256") != baseline_signature.get("sha256")
    diff_nonzero = int(diff_signature.get("nonzero", 0) or 0)

    return {
        "i": frame_index,
        "data_index": data_index,
        "frame_idx": probe_record.get("frame_idx"),
        "cam_token": cam_token,
        "scene_token": scene_token,
        "geometry_metadata": metadata,
        "operation": operation,
        "baseline_signature": baseline_signature,
        "converter_signature": converter_signature,
        "baseline_matches_converter_signature": baseline_matches_converter,
        "candidate_raster_path": str(candidate_path),
        "candidate_signature": candidate_signature,
        "diff_path": str(diff_path),
        "diff_signature": diff_signature,
        "candidate_differs_from_baseline": candidate_differs_from_baseline,
        "diff_nonzero": diff_nonzero,
        "claim_boundary": {
            "dry_run_candidate_is_synthetic": True,
            "dry_run_candidate_is_not_verified_lane_geometry_replacement": True,
            "raster_diff_is_not_lane_change_control": True,
            "does_not_run_gpu": True,
        },
    }


def build_summary(
    records: list[dict[str, Any]],
    probe_path: Path,
    raw_root: str,
    dx_pixels: float,
    dy_pixels: float,
) -> dict[str, Any]:
    baseline_match_true = sum(1 for record in records if record.get("baseline_matches_converter_signature") is True)
    baseline_match_false = sum(1 for record in records if record.get("baseline_matches_converter_signature") is False)
    candidate_changed_true = sum(1 for record in records if record.get("candidate_differs_from_baseline") is True)
    candidate_changed_false = sum(1 for record in records if record.get("candidate_differs_from_baseline") is False)
    total_diff_nonzero = sum(int(record.get("diff_nonzero", 0) or 0) for record in records)
    total_modified_lane_dividers = sum(
        int((record.get("operation") or {}).get("modified_visible_count", 0) or 0)
        for record in records
    )

    return {
        "schema_version": "candidate70_hdmap_lane_divider_dry_run_builder.v0",
        "audit_only": True,
        "does_not_run_gpu": True,
        "does_not_generate_video": True,
        "does_not_modify_model_inputs": True,
        "probe_path": str(probe_path),
        "raw_root": raw_root,
        "operation": {
            "operation": "translate_projected_lane_divider_pixels",
            "target_type_name": "lane_divider",
            "dx_pixels": dx_pixels,
            "dy_pixels": dy_pixels,
            "synthetic_dry_run": True,
        },
        "frame_count": len(records),
        "baseline_match_true": baseline_match_true,
        "baseline_match_false": baseline_match_false,
        "candidate_changed_true": candidate_changed_true,
        "candidate_changed_false": candidate_changed_false,
        "total_diff_nonzero": total_diff_nonzero,
        "total_modified_visible_lane_dividers": total_modified_lane_dividers,
        "records": records,
        "claim": {
            "candidate70_lane_divider_dry_run_candidate_built": bool(records),
            "candidate70_dry_run_raster_diff_observed": bool(records) and candidate_changed_false == 0 and total_diff_nonzero > 0,
            "candidate70_true_lane_geometry_replacement_available": False,
            "hdmap_lane_geometry_override_verified": False,
            "lane_change_control_verified": False,
            "runtime_motion_control_connected": False,
            "semantic_success_claim_allowed": False,
        },
        "claim_boundary": {
            "dry_run_candidate_is_synthetic_not_verified_map_geometry": True,
            "lane_divider_pixel_translation_is_not_lane_change_control": True,
            "hdmap_raster_diff_is_not_video_semantic_success": True,
            "gpu_requires_separate_readiness_gate": True,
        },
        "next_required_steps": [
            "Inspect dry-run candidate rasters and diffs before considering any replacement source claim.",
            "Define a map-geometry-grounded replacement operation before claiming true lane geometry replacement.",
            "Run GPU only after separate readiness gate and explicit user approval.",
        ],
    }


def run_builder(
    probe_path: Path,
    output_path: Path,
    images_dir: Path,
    dx_pixels: float,
    dy_pixels: float,
    max_frames: int | None,
) -> dict[str, Any]:
    probe = load_json(probe_path)
    raw_root = str(probe.get("raw_root", "/data/projects/DriveLoop/data/raw/nuscenes"))
    records = probe.get("records", [])
    if not isinstance(records, list) or not records:
        raise SystemExit(f"No records found in probe: {probe_path}")

    version = "v1.0-mini"
    dataroot = Path(raw_root)
    if dataroot.name == version:
        dataroot = dataroot.parent
    converter = NuScenesConverter(
        data_dir=str(dataroot),
        version=version,
        save_path="/tmp/driveloop_candidate70_hdmap_lane_divider_dry_run_unused",
        save_version="v0.0.1",
    )

    selected_records = records[:max_frames] if max_frames is not None else records
    frame_records = [
        build_frame_record(
            converter=converter,
            probe_record=record,
            images_dir=images_dir,
            dx_pixels=dx_pixels,
            dy_pixels=dy_pixels,
        )
        for record in selected_records
        if isinstance(record, dict)
    ]
    summary = build_summary(
        records=frame_records,
        probe_path=probe_path,
        raw_root=raw_root,
        dx_pixels=dx_pixels,
        dy_pixels=dy_pixels,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a non-GPU candidate70 lane-divider dry-run raster.")
    parser.add_argument("--probe-path", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--dx-pixels", type=float, default=-32.0)
    parser.add_argument("--dy-pixels", type=float, default=0.0)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    summary = run_builder(
        probe_path=args.probe_path,
        output_path=args.output,
        images_dir=args.images_dir,
        dx_pixels=args.dx_pixels,
        dy_pixels=args.dy_pixels,
        max_frames=args.max_frames,
    )
    print(args.output)
    print(json.dumps({
        "schema_version": summary["schema_version"],
        "frame_count": summary["frame_count"],
        "baseline_match_true": summary["baseline_match_true"],
        "baseline_match_false": summary["baseline_match_false"],
        "candidate_changed_true": summary["candidate_changed_true"],
        "candidate_changed_false": summary["candidate_changed_false"],
        "total_diff_nonzero": summary["total_diff_nonzero"],
        "total_modified_visible_lane_dividers": summary["total_modified_visible_lane_dividers"],
        "claim": summary["claim"],
        "claim_boundary": summary["claim_boundary"],
    }, indent=2))


if __name__ == "__main__":
    main()
