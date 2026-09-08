from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_LABELS = Path(
    "outputs/driveloop/tiny_real_actor_identity_runtime_dataset/"
    "cam_front_8/v0.0.1/labels/data.pkl"
)
DEFAULT_OUTPUT = Path(
    "outputs/driveloop/actor_track_surface_audit/"
    "tiny_real_actor_track_surface_audit.json"
)


def load_label_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [item for item in data.values() if isinstance(item, dict)]
    return []


def list_shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    return list(shape) if shape is not None else None


def to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value if isinstance(value, list) else []


def numeric_preview(value: Any, limit: int | None = None) -> list[Any] | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        return None
    items = value[:limit] if limit is not None else value
    return [round(float(item), 4) if isinstance(item, (int, float)) else item for item in items]


def same_sequence_rows(rows: list[dict[str, Any]], max_frames: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    first = rows[0]
    scene = first.get("scene_token")
    cam = first.get("cam_type")
    selected = [
        item
        for item in rows
        if item.get("scene_token") == scene and item.get("cam_type") == cam
    ]
    return sorted(selected, key=lambda x: x.get("frame_idx", -1))[:max_frames]


def is_valid_instance_token(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and value.strip() in {"", "None", "null"}:
        return False
    return True


def build_actor_track_surface_audit(
    labels_path: Path,
    max_frames: int = 8,
    max_tracks_preview: int = 8,
) -> dict[str, Any]:
    rows = load_label_rows(labels_path)
    sequence = same_sequence_rows(rows, max_frames=max_frames)

    tracks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    frame_summaries = []
    mismatches = []

    for item in sequence:
        boxes = to_list(item.get("boxes3d"))
        velocities = to_list(item.get("velocities"))
        instance_tokens = to_list(item.get("instance_tokens"))
        ann_tokens = to_list(item.get("sample_annotation_tokens"))
        categories = to_list(item.get("actor_identity_categories"))
        labels = to_list(item.get("ori_labels3d", item.get("labels3d")))

        box_count = len(boxes)
        instance_count = len(instance_tokens)
        ann_count = len(ann_tokens)

        if instance_tokens and instance_count != box_count:
            mismatches.append(
                {
                    "frame_idx": item.get("frame_idx"),
                    "field": "instance_tokens",
                    "expected_box_count": box_count,
                    "actual_count": instance_count,
                }
            )
        if ann_tokens and ann_count != box_count:
            mismatches.append(
                {
                    "frame_idx": item.get("frame_idx"),
                    "field": "sample_annotation_tokens",
                    "expected_box_count": box_count,
                    "actual_count": ann_count,
                }
            )

        frame_summaries.append(
            {
                "frame_idx": item.get("frame_idx"),
                "data_index": item.get("data_index"),
                "sample_token": item.get("sample_token"),
                "cam_token": item.get("cam_token"),
                "boxes3d_shape": list_shape(item.get("boxes3d")),
                "velocities_shape": list_shape(item.get("velocities")),
                "instance_tokens_count": instance_count,
                "sample_annotation_tokens_count": ann_count,
            }
        )

        for box_index, instance_token in enumerate(instance_tokens):
            if box_index >= box_count or not is_valid_instance_token(instance_token):
                continue
            box = boxes[box_index]
            velocity = velocities[box_index] if box_index < len(velocities) else None
            tracks[str(instance_token)].append(
                {
                    "frame_idx": item.get("frame_idx"),
                    "data_index": item.get("data_index"),
                    "sample_token": item.get("sample_token"),
                    "box_index": box_index,
                    "sample_annotation_token": ann_tokens[box_index] if box_index < len(ann_tokens) else None,
                    "category": categories[box_index] if box_index < len(categories) else (
                        labels[box_index] if box_index < len(labels) else None
                    ),
                    "box3d": numeric_preview(box),
                    "center_xyz": numeric_preview(box, limit=3),
                    "velocity_xy": numeric_preview(velocity),
                }
            )

    track_items = sorted(
        tracks.items(),
        key=lambda pair: (-len(pair[1]), pair[0]),
    )
    persistent_tracks = [
        (token, observations)
        for token, observations in track_items
        if len({obs.get("frame_idx") for obs in observations}) >= 2
    ]

    tracks_preview = []
    for token, observations in track_items[:max_tracks_preview]:
        frame_indices = [obs.get("frame_idx") for obs in observations]
        tracks_preview.append(
            {
                "instance_token": token,
                "observation_count": len(observations),
                "frame_indices": frame_indices,
                "category": observations[0].get("category") if observations else None,
                "first_observation": observations[0] if observations else None,
                "last_observation": observations[-1] if observations else None,
            }
        )

    identity_available = bool(track_items)
    boxes_grouped_by_identity = bool(persistent_tracks) and not mismatches
    status = "per_frame_actor_tracks_observed" if boxes_grouped_by_identity else "not_observed"

    blockers = []
    if not identity_available:
        blockers.append("actor_identity_not_available")
    if mismatches:
        blockers.append("identity_box_count_mismatch")
    if not persistent_tracks:
        blockers.append("persistent_actor_tracks_not_observed_across_frames")

    return {
        "schema_version": "driveloop_actor_track_surface_audit.v0",
        "status": status,
        "inputs": {
            "labels_path": str(labels_path),
            "max_frames": max_frames,
        },
        "dataset_surface": {
            "rows_available": len(rows),
            "sequence_frames_inspected": len(sequence),
            "sequence_scene_token": sequence[0].get("scene_token") if sequence else None,
            "sequence_cam_type": sequence[0].get("cam_type") if sequence else None,
            "frames": frame_summaries,
            "mismatches": mismatches,
        },
        "track_surface": {
            "actor_identity_available": identity_available,
            "boxes_grouped_by_instance_token": boxes_grouped_by_identity,
            "track_count": len(track_items),
            "persistent_track_count": len(persistent_tracks),
            "max_track_length": max((len(obs) for _, obs in track_items), default=0),
            "tracks_preview": tracks_preview,
        },
        "claim": {
            "per_frame_actor_identity_observed": identity_available,
            "per_frame_actor_boxes3d_grouped_by_identity": boxes_grouped_by_identity,
            "runtime_motion_control_connected": False,
            "lane_change_control_verified": False,
            "semantic_success_claim_allowed": False,
        },
        "blockers": blockers,
        "claim_boundary": {
            "actor_track_audit_is_not_runtime_motion_control": True,
            "actor_track_audit_is_not_video_semantic_success": True,
            "grouped_boxes_do_not_prove_lane_change_control": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit per-frame actor track surfaces from processed labels.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-frames", type=int, default=8)
    args = parser.parse_args()

    report = build_actor_track_surface_audit(args.labels, max_frames=args.max_frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
