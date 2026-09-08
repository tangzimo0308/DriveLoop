from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_LABELS = Path("/data/projects/DriveLoop/data/processed/nuscenes/v1.0-mini/cam_all_val/v0.0.2/labels/data.pkl")
CAM_NAMES = ["CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"]


def normalize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
            "sum": float(value.astype(float).sum()) if value.size else 0.0,
        }
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    return value


def digest(value: Any) -> str:
    payload = json.dumps(normalize(value), sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = None
        for key in ("data", "infos", "records", "samples"):
            if isinstance(payload.get(key), list):
                records = payload[key]
                break
        if records is None:
            raise ValueError(f"Unsupported pickle dict keys: {sorted(payload.keys())[:20]}")
    else:
        raise TypeError(f"Unsupported pickle payload type: {type(payload)!r}")

    return [item for item in records if isinstance(item, dict)]


def is_front_record(record: dict[str, Any]) -> bool:
    return str(record.get("cam_type", "")).lower() == "cam_front"


def is_video_start(record: dict[str, Any], frame_num: int, hz_factor: int, video_split_rate: int) -> bool:
    frame_idx = int(record.get("frame_idx", -1))
    video_length = int(record.get("video_length", -1))
    video_frame_len = hz_factor * frame_num
    split = max(1, video_frame_len // max(1, video_split_rate))
    return frame_idx >= 0 and video_length >= 0 and frame_idx % split == 0 and frame_idx + video_frame_len <= video_length


def candidate_camera_starts(
    records: list[dict[str, Any]],
    *,
    frame_num: int,
    hz_factor: int,
    video_split_rate: int,
    multiview: bool,
) -> list[list[int]]:
    starts: list[list[int]] = []

    for idx, record in enumerate(records):
        if not is_front_record(record):
            continue
        if not is_video_start(record, frame_num=frame_num, hz_factor=hz_factor, video_split_rate=video_split_rate):
            continue

        if not multiview:
            starts.append([idx])
            continue

        mv = record.get("multiview_start_idx", {})
        if not isinstance(mv, dict):
            continue
        try:
            group = [int(mv[name]) for name in CAM_NAMES]
        except Exception:
            continue
        group.insert(1, idx)
        starts.append(group)

    return starts


def selected_frame_indices(camera_starts: list[int], *, frame_num: int, hz_factor: int) -> list[int]:
    arr = np.array(camera_starts, dtype=np.int64)
    return np.stack([arr + i * hz_factor for i in range(frame_num)]).T.reshape(-1).astype(int).tolist()


def label_counter(records: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        labels = record.get("labels3d", [])
        for label in labels:
            if isinstance(label, (list, tuple)):
                counter[".".join(str(part) for part in label)] += 1
            else:
                counter[str(label)] += 1
    return dict(sorted(counter.items()))


def summarize_candidate(
    records: list[dict[str, Any]],
    *,
    candidate_index: int,
    camera_starts: list[int],
    frame_num: int,
    hz_factor: int,
) -> dict[str, Any]:
    selected = selected_frame_indices(camera_starts, frame_num=frame_num, hz_factor=hz_factor)
    selected_records = [records[i] for i in selected if 0 <= i < len(records)]
    front_record = records[camera_starts[1] if len(camera_starts) > 1 else camera_starts[0]]

    scene_descriptions = sorted({str(item.get("scene_description")) for item in selected_records})
    sample_tokens = sorted({str(item.get("sample_token")) for item in selected_records})
    scene_tokens = sorted({str(item.get("scene_token")) for item in selected_records})

    return {
        "candidate_index": candidate_index,
        "camera_start_indices": camera_starts,
        "selected_frame_indices_preview": selected[:18],
        "selected_record_count": len(selected_records),
        "front_record": {
            "index": camera_starts[1] if len(camera_starts) > 1 else camera_starts[0],
            "frame_idx": front_record.get("frame_idx"),
            "video_length": front_record.get("video_length"),
            "cam_type": front_record.get("cam_type"),
            "scene_description": front_record.get("scene_description"),
            "sample_token": front_record.get("sample_token"),
            "scene_token": front_record.get("scene_token"),
        },
        "unique_scene_descriptions": scene_descriptions[:10],
        "unique_sample_token_count": len(sample_tokens),
        "unique_scene_token_count": len(scene_tokens),
        "label_counts": label_counter(selected_records),
        "signatures": {
            "scene_descriptions": digest(scene_descriptions),
            "labels3d": digest([item.get("labels3d") for item in selected_records]),
            "boxes3d": digest([item.get("boxes3d") for item in selected_records]),
            "sample_tokens": digest(sample_tokens),
            "selected_indices": digest(selected),
        },
    }


def build_report(
    labels_path: Path,
    *,
    max_skip: int,
    frame_num: int,
    cam_num: int,
    hz_factor: int,
    video_split_rate: int,
    multiview: bool,
) -> dict[str, Any]:
    records = load_records(labels_path)
    starts = candidate_camera_starts(
        records,
        frame_num=frame_num,
        hz_factor=hz_factor,
        video_split_rate=video_split_rate,
        multiview=multiview,
    )

    limit = min(max_skip + 1, len(starts))
    candidates = [
        summarize_candidate(
            records,
            candidate_index=i,
            camera_starts=starts[i],
            frame_num=frame_num,
            hz_factor=hz_factor,
        )
        for i in range(limit)
    ]

    return {
        "schema_version": "driveloop_dd2_batch_sampler_audit.v0",
        "labels_path": str(labels_path),
        "record_count": len(records),
        "candidate_start_count": len(starts),
        "config": {
            "frame_num": frame_num,
            "cam_num": cam_num,
            "hz_factor": hz_factor,
            "video_split_rate": video_split_rate,
            "multiview": multiview,
        },
        "candidate_batch_semantics": "candidate_index corresponds to dd2 batch skip when shuffle is disabled or deterministic order is used",
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only audit of DD2 candidate baseline batches.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--max-skip", type=int, default=5)
    parser.add_argument("--frame-num", type=int, default=8)
    parser.add_argument("--cam-num", type=int, default=6)
    parser.add_argument("--hz-factor", type=int, default=3)
    parser.add_argument("--video-split-rate", type=int, default=1)
    parser.add_argument("--single-view", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = build_report(
        args.labels,
        max_skip=args.max_skip,
        frame_num=args.frame_num,
        cam_num=args.cam_num,
        hz_factor=args.hz_factor,
        video_split_rate=args.video_split_rate,
        multiview=not args.single_view,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
