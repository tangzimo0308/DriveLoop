from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path("outputs/driveloop/runtime_surface_code_audit/motorcycle_refined_runtime_surface_code_audit.json")


SOURCE_PATHS = {
    "converter": Path("dreamer-datasets/dd_scripts/converters/nuscenes_converter.py"),
    "config": Path("dreamer-train/projects/DriveDreamer2/configs/drivedreamer2_img_cond_mini_local.py"),
    "transform": Path("dreamer-train/projects/DriveDreamer2/drivedreamer2/drivedreamer2_transforms.py"),
    "tester": Path("dreamer-train/projects/DriveDreamer2/drivedreamer2/drivedreamer2_tester.py"),
    "pipeline": Path("dreamer-models/dreamer_models/pipelines/drivedreamer2/pipeline_drivedreamer2.py"),
    "unet": Path("dreamer-models/dreamer_models/models/drivedreamer2/unet_spatio_temporal_condition.py"),
}


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def first_line(path: Path, needle: str) -> dict[str, Any]:
    text = read_text(path)
    for idx, line in enumerate(text.splitlines(), start=1):
        if needle in line:
            return {"path": str(path), "line": idx, "text": line.strip()}
    return {"path": str(path), "line": None, "text": None}


def has_all(path: Path, needles: list[str]) -> bool:
    text = read_text(path)
    return all(needle in text for needle in needles)


def build_runtime_surface_code_audit(repo_root: Path = Path(".")) -> dict[str, Any]:
    paths = {name: repo_root / path for name, path in SOURCE_PATHS.items()}

    converter_has_velocity = has_all(paths["converter"], ["box_velocity", "'velocities': velocities"])
    converter_has_lane_hdmap = has_all(paths["converter"], ["'lane'", "image_hdmap", "get_map_geom"])
    config_uses_raster_inputs = has_all(paths["config"], ["gd_input_name='image_hdmap'", "bd_input_name='image_box'"])
    transform_builds_static_box_canvas = has_all(paths["transform"], ["data_dict['boxes3d']", "generate_canvas_box", "new_data_dict['box_downsampler_input']"])
    tester_runtime_keys_are_downsamplers = has_all(
        paths["tester"],
        [
            "'grounding_downsampler_input': grounding_downsampler_input",
            "'box_downsampler_input': box_downsampler_input",
            '"grounding_downsampler_input": tensor_summary',
            '"box_downsampler_input": tensor_summary',
        ],
    )
    pipeline_consumes_downsamplers = has_all(
        paths["pipeline"],
        [
            "input_dict.get('grounding_downsampler_input'",
            "input_dict.get('box_downsampler_input'",
            "self.grounding_downsampler(grounding_downsampler_input)",
            "self.box_downsampler(box_downsampler_input)",
        ],
    )
    unet_consumes_condition_latents = has_all(
        paths["unet"],
        [
            'input_dict["grounding_downsampler_latents"]',
            'input_dict["box_downsampler_latents"]',
            "torch.cat([sample, grounding_downsampler_latents,box_downsampler_latents]",
        ],
    )

    runtime_text = "\n".join(read_text(path) for name, path in paths.items() if name in {"tester", "pipeline", "unet"})
    direct_motion_runtime_patterns = [
        "input_dict.get('trajectory",
        'input_dict.get("trajectory',
        "input_dict.get('trajectories",
        'input_dict.get("trajectories',
        "input_dict.get('actor_trajectory",
        'input_dict.get("actor_trajectory',
        "input_dict.get('future_trajectory",
        'input_dict.get("future_trajectory',
        "input_dict.get('displacement",
        'input_dict.get("displacement',
        "input_dict.get('actor_displacement",
        'input_dict.get("actor_displacement',
        "input_dict.get('velocity",
        'input_dict.get("velocity',
        "input_dict.get('velocities",
        'input_dict.get("velocities',
        "input_dict['trajectory",
        'input_dict["trajectory',
        "input_dict['velocity",
        'input_dict["velocity',
        "'trajectory':",
        '"trajectory":',
        "'velocity':",
        '"velocity":',
        "'velocities':",
        '"velocities":',
    ]
    metadata_only_motion_terms = [
        "motion_metadata",
        "velocities_available_in_batch",
        "velocities_shape",
    ]
    runtime_has_direct_motion_surface = any(pattern in runtime_text for pattern in direct_motion_runtime_patterns)
    runtime_has_motion_metadata_only = any(term in runtime_text for term in metadata_only_motion_terms)

    return {
        "schema_version": "driveloop_runtime_surface_code_audit.v0",
        "status": "not_runtime_connected",
        "status_reason": "converter exposes velocity and lane/HDMap source data, but DD2 runtime consumes only image_hdmap/image_box downsampler surfaces plus image/video condition",
        "does_not_run_gpu": True,
        "semantic_success_claim_allowed": False,
        "surfaces": {
            "dataset_velocity": {
                "status": "available_in_converter" if converter_has_velocity else "not_observed",
                "evidence": [
                    first_line(paths["converter"], "box_velocity"),
                    first_line(paths["converter"], "'velocities': velocities"),
                ],
            },
            "dataset_lane_hdmap": {
                "status": "rasterized_image_hdmap_from_lane_geometry" if converter_has_lane_hdmap else "not_observed",
                "evidence": [
                    first_line(paths["converter"], "get_map_geom"),
                    first_line(paths["converter"], "image_hdmap"),
                ],
            },
            "runtime_condition_inputs": {
                "status": "image_hdmap_and_image_box_downsamplers" if config_uses_raster_inputs else "unknown",
                "evidence": [
                    first_line(paths["config"], "gd_input_name='image_hdmap'"),
                    first_line(paths["config"], "bd_input_name='image_box'"),
                ],
            },
            "static_box_canvas": {
                "status": "observed" if transform_builds_static_box_canvas else "unknown",
                "interpretation": "static/spatial box raster input; not temporal actor motion control",
                "evidence": [
                    first_line(paths["transform"], "data_dict['boxes3d']"),
                    first_line(paths["transform"], "generate_canvas_box"),
                    first_line(paths["transform"], "new_data_dict['box_downsampler_input']"),
                ],
            },
            "dd2_runtime_input_dict": {
                "status": "downsamplers_only" if tester_runtime_keys_are_downsamplers else "unknown",
                "evidence": [
                    first_line(paths["tester"], "'grounding_downsampler_input': grounding_downsampler_input"),
                    first_line(paths["tester"], "'box_downsampler_input': box_downsampler_input"),
                ],
            },
            "pipeline_consumption": {
                "status": "downsampler_latents_consumed" if pipeline_consumes_downsamplers else "unknown",
                "evidence": [
                    first_line(paths["pipeline"], "self.grounding_downsampler(grounding_downsampler_input)"),
                    first_line(paths["pipeline"], "self.box_downsampler(box_downsampler_input)"),
                ],
            },
            "unet_consumption": {
                "status": "condition_latents_concatenated" if unet_consumes_condition_latents else "unknown",
                "evidence": [
                    first_line(paths["unet"], 'input_dict["grounding_downsampler_latents"]'),
                    first_line(paths["unet"], 'input_dict["box_downsampler_latents"]'),
                ],
            },
            "direct_motion_runtime_surface": {
                "status": "not_observed" if not runtime_has_direct_motion_surface else "needs_manual_review",
                "interpretation": "velocity mentions observed in DD2 runtime are metadata-only unless a trajectory/velocity/displacement tensor is passed through input_dict and consumed by pipeline/UNet",
                "runtime_consumption_patterns_checked": direct_motion_runtime_patterns,
                "metadata_only_motion_terms_observed": runtime_has_motion_metadata_only,
                "searched_terms": [
                    "trajectory_tensor",
                    "actor_trajectory",
                    "future_trajectory",
                    "actor_displacement",
                    "track_id",
                    "velocities",
                    "velocity_tensor",
                ],
            },
        },
        "claim_boundary": {
            "dataset_velocity_is_not_runtime_motion_control": True,
            "image_box_canvas_is_not_temporal_motion_control": True,
            "image_hdmap_raster_is_not_verified_lane_geometry_override": True,
            "runtime_tensor_audit_is_not_video_semantic_success": True,
            "semantic_success_requires_measured_passed_review": True,
        },
        "next_required_steps": [
            "record this as negative runtime-surface evidence",
            "do not claim lane-change control from static image_box or image_hdmap surfaces",
            "if adding a motion intervention, first expose an audit-only runtime tensor or metadata path for velocity/trajectory/per-frame boxes",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit DD2 code paths for DriveLoop runtime control surfaces.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    audit = build_runtime_surface_code_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(args.output)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
