from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from driveloop.dd2_override import read_override_audit
from scripts.run_dd2_structural_audit import load_config, tensor_sig, transform_first_sample


DEFAULT_CONFIG = "dreamer-train/projects/DriveDreamer2/configs/drivedreamer2_img_cond_mini_local.py"
DEFAULT_OUTPUT = "outputs/driveloop/hdmap_runtime_surface_audit/mini_hdmap_zero_surface_audit.json"


def signature_changed(baseline: dict[str, Any], zeroed: dict[str, Any], key: str) -> bool:
    return baseline.get(key, {}).get("sha256") != zeroed.get(key, {}).get("sha256")


def build_report(
    baseline_signatures: dict[str, Any],
    zero_signatures: dict[str, Any],
    override_audit: dict[str, Any],
) -> dict[str, Any]:
    grounding_changed = signature_changed(
        baseline_signatures,
        zero_signatures,
        "grounding_downsampler_input",
    )
    box_changed = signature_changed(
        baseline_signatures,
        zero_signatures,
        "box_downsampler_input",
    )
    image_changed = signature_changed(
        baseline_signatures,
        zero_signatures,
        "input_image",
    )
    changed_counts = override_audit.get("changed_counts", {})
    hdmap_override_changed = bool(changed_counts.get("image_hdmap"))

    status = (
        "hdmap_raster_runtime_surface_mutable"
        if grounding_changed and hdmap_override_changed
        else "not_observed"
    )

    return {
        "schema_version": "driveloop_hdmap_runtime_surface_audit.v0",
        "status": status,
        "does_not_run_gpu": True,
        "does_not_generate_video": True,
        "intervention": {
            "target": "image_hdmap",
            "mode": "zero",
            "source": "explicit_hdmap_raster_zero_ablation",
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
                "zero_ablation": zero_signatures.get("grounding_downsampler_input"),
                "interpretation": "HDMap raster reaches the DD2 grounding downsampler runtime surface.",
            },
            "box_downsampler_input": {
                "changed": box_changed,
                "interpretation": "Expected to remain unchanged for an HDMap-only ablation.",
            },
            "input_image": {
                "changed": image_changed,
                "interpretation": "Expected to remain unchanged for a structural-condition ablation.",
            },
        },
        "claim": {
            "hdmap_raster_runtime_surface_mutable": status == "hdmap_raster_runtime_surface_mutable",
            "hdmap_lane_geometry_override_verified": False,
            "lane_change_control_verified": False,
            "runtime_motion_control_connected": False,
            "video_semantic_claim": "not_evaluated_by_this_audit",
            "semantic_success_claim_allowed": False,
        },
        "claim_boundary": {
            "zero_hdmap_ablation_is_not_lane_geometry_override": True,
            "hdmap_raster_hash_change_is_not_lane_change_control": True,
            "runtime_tensor_audit_is_not_video_semantic_success": True,
            "semantic_success_requires_measured_passed_review": True,
        },
        "next_required_steps": [
            "Record HDMap raster as a mutable DD2 runtime surface if grounding_downsampler_input changed.",
            "Do not claim lane geometry override from zero ablation.",
            "Audit lane geometry compatibility or a verified replacement HDMap source before any lane-change intervention.",
        ],
    }


def run_audit(config_path: Path, output_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    data_cfg = config["dataloaders"]["test"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    override_audit_path = output_path.with_suffix(".override_audit.jsonl")

    override_json = {
        "available": True,
        "schema_version": "driveloop_dd2_override.v0",
        "source": "hdmap_runtime_surface_audit",
        "image_hdmap": {
            "mode": "zero",
            "source": "explicit_hdmap_raster_zero_ablation",
        },
        "boxes3d": {
            "mode": "append",
            "append": [],
            "source": "not_modified",
        },
        "audit": {
            "control_level": "runtime_surface_ablation",
            "limitations": [
                "zero_hdmap_ablation_is_not_lane_geometry_override",
                "hdmap_raster_change_does_not_prove_lane_change_control",
            ],
        },
    }

    try:
        baseline = transform_first_sample(data_cfg, override_json=None, audit_path=None)
        zeroed = transform_first_sample(data_cfg, override_json=override_json, audit_path=override_audit_path)

        keys = ["grounding_downsampler_input", "box_downsampler_input", "input_image"]
        baseline_signatures = {key: tensor_sig(baseline[key]) for key in keys}
        zero_signatures = {key: tensor_sig(zeroed[key]) for key in keys}
        override_audit = read_override_audit(override_audit_path)

        report = build_report(
            baseline_signatures=baseline_signatures,
            zero_signatures=zero_signatures,
            override_audit=override_audit,
        )
        report["inputs"] = {
            "config_path": str(config_path),
            "override_audit_path": str(override_audit_path),
        }
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        os.environ.pop("DRIVELOOP_DD2_OVERRIDE_JSON", None)
        os.environ.pop("DRIVELOOP_DD2_OVERRIDE_AUDIT_PATH", None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether image_hdmap raster reaches the DD2 grounding runtime surface."
    )
    parser.add_argument("--config-path", type=Path, default=Path(DEFAULT_CONFIG))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    args = parser.parse_args()

    report = run_audit(args.config_path, args.output)
    print(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
