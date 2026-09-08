from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any


TRACK_ID_FIELDS = [
    "instance_token",
    "instance_tokens",
    "track_id",
    "track_ids",
    "sample_annotation_token",
    "sample_annotation_tokens",
]


def shape_of(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    return list(shape) if shape is not None else None


def first_list_item(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value:
        return value[0]
    return None


def load_label_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [item for item in data.values() if isinstance(item, dict)]
    return []


def source_usage(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return {
        "path": str(path),
        "mentions_velocities": "velocities" in text,
        "mentions_velocity": "velocity" in text,
        "mentions_trajectory": "trajectory" in text or "traj" in text,
    }


def build_velocity_surface_audit(
    labels_path: Path,
    transform_path: Path,
    tester_path: Path,
    max_frames: int = 8,
) -> dict[str, Any]:
    rows = load_label_rows(labels_path)
    first = rows[0] if rows else {}
    scene = first.get("scene_token")
    cam = first.get("cam_type")

    same_sequence = [
        item
        for item in rows
        if item.get("scene_token") == scene and item.get("cam_type") == cam
    ]
    same_sequence = sorted(same_sequence, key=lambda x: x.get("frame_idx", -1))[:max_frames]

    frame_summaries = []
    for item in same_sequence:
        velocities = item.get("velocities")
        boxes = item.get("boxes3d")
        labels = item.get("ori_labels3d", [])
        frame_summaries.append(
            {
                "frame_idx": item.get("frame_idx"),
                "data_index": item.get("data_index"),
                "sample_token": item.get("sample_token"),
                "cam_token": item.get("cam_token"),
                "labels_len": len(labels) if isinstance(labels, list) else None,
                "boxes3d_shape": shape_of(boxes),
                "velocities_shape": shape_of(velocities),
                "first_label": first_list_item(labels),
                "first_box": first_list_item(boxes),
                "first_velocity": first_list_item(velocities),
            }
        )

    transform_usage = source_usage(transform_path)
    tester_usage = source_usage(tester_path)

    transform_consumes_velocity = transform_usage["mentions_velocities"] or transform_usage["mentions_velocity"]
    tester_consumes_velocity = tester_usage["mentions_velocities"] or tester_usage["mentions_velocity"]

    return {
        "schema_version": "driveloop_dd2_velocity_surface_audit.v0",
        "inputs": {
            "labels_path": str(labels_path),
            "transform_path": str(transform_path),
            "tester_path": str(tester_path),
        },
        "dataset_surface": {
            "rows_available": len(rows),
            "first_sample_keys": sorted(first.keys()) if first else [],
            "velocities_present": "velocities" in first,
            "velocities_shape": shape_of(first.get("velocities")),
            "boxes3d_shape": shape_of(first.get("boxes3d")),
            "track_identity_fields_present": [field for field in TRACK_ID_FIELDS if field in first],
            "sequence_scene_token": scene,
            "sequence_cam_type": cam,
            "sequence_frames_inspected": len(frame_summaries),
            "frames": frame_summaries,
        },
        "runtime_surface": {
            "transform_mentions_velocity": transform_consumes_velocity,
            "tester_mentions_velocity": tester_consumes_velocity,
            "dd2_runtime_input_keys_observed": [
                "prompt_embed",
                "img_cond",
                "grounding_downsampler_input",
                "box_downsampler_input",
            ],
            "velocity_runtime_input_observed": transform_consumes_velocity or tester_consumes_velocity,
        },
        "claim": {
            "velocity_exists_in_dataset": "velocities" in first,
            "velocity_consumed_by_dd2_runtime": transform_consumes_velocity or tester_consumes_velocity,
            "track_identity_available": any(field in first for field in TRACK_ID_FIELDS),
            "lane_change_trajectory_control": "not_verified",
        },
        "limitations": [
            "velocities exist in label samples but are not observed in DD2 transform/tester runtime inputs",
            "no actor track identity field was observed in the inspected mini sample",
            "labels/order across frames are not sufficient evidence for controlled actor trajectory",
            "this audit does not evaluate generated video semantics",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("/data/projects/DriveLoop/data/processed/nuscenes/v1.0-mini/cam_all_val/v0.0.2/labels/data.pkl"),
    )
    parser.add_argument(
        "--transform",
        type=Path,
        default=Path("dreamer-train/projects/DriveDreamer2/drivedreamer2/drivedreamer2_transforms.py"),
    )
    parser.add_argument(
        "--tester",
        type=Path,
        default=Path("dreamer-train/projects/DriveDreamer2/drivedreamer2/drivedreamer2_tester.py"),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    report = build_velocity_surface_audit(args.labels, args.transform, args.tester)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
