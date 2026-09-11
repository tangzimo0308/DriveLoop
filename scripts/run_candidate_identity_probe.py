#!/usr/bin/env python3
"""Emit an identity summary for a new source-candidate window.

The summary is the token source consumed by source-sample binding and
by the runtime subset builder (build_candidate70_runtime_subset).
Selection is double-anchored: the candidate index under the pinned
enumeration parameters AND the expected frame-0 sample token must both
match, so a future enumeration drift fails loudly instead of probing
the wrong window. Target presence is verified per frame from the
converted records' instance tokens; a missing target in any frame is
a hard failure.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from scripts.run_dd2_batch_sampler_audit import candidate_camera_starts


def load_records(path: Path) -> list:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return [record for record in payload if isinstance(record, dict)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Emit an identity summary for a candidate window.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--candidate-index", type=int, required=True)
    parser.add_argument("--expect-f0-sample-token", required=True)
    parser.add_argument("--instance-token", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frame-num", type=int, default=8)
    parser.add_argument("--hz-factor", type=int, default=3)
    parser.add_argument("--video-split-rate", type=int, default=1)
    parser.add_argument("--single-view", action="store_true")
    parser.add_argument("--raw-root", default="/data/projects/DriveLoop/data/raw/nuscenes")
    args = parser.parse_args(argv)

    records = load_records(Path(args.dataset_dir) / "labels" / "data.pkl")
    starts = candidate_camera_starts(
        records,
        frame_num=args.frame_num,
        hz_factor=args.hz_factor,
        video_split_rate=args.video_split_rate,
        multiview=not args.single_view,
    )
    if not 0 <= args.candidate_index < len(starts):
        parser.error(
            "candidate index %d out of range (%d candidates)"
            % (args.candidate_index, len(starts))
        )
    group = starts[args.candidate_index]
    front = group[0] if args.single_view else group[1]

    frame_summaries = []
    all_instance = all_annotation = all_target = True
    for step in range(args.frame_num):
        record = records[front + step * args.hz_factor]
        labels = list(record.get("ori_labels3d", []))
        tokens = record.get("instance_tokens") or []
        annotations = record.get("sample_annotation_tokens") or []
        target_indices = [
            index for index, token in enumerate(tokens)
            if str(token) == args.instance_token
        ]
        target_annotations = [
            str(annotations[index]) for index in target_indices
            if index < len(annotations)
        ]
        all_instance &= bool(tokens) and len(tokens) == len(labels)
        all_annotation &= bool(annotations) and len(annotations) == len(labels)
        all_target &= bool(target_indices)
        frame_summaries.append({
            "data_index": record.get("data_index"),
            "frame_idx": record.get("frame_idx"),
            "cam_token": record.get("cam_token"),
            "sample_token": record.get("sample_token"),
            "box_count": len(labels),
            "instance_token_count": len(tokens),
            "sample_annotation_token_count": len(annotations),
            "target_present": bool(target_indices),
            "target_box_indices": target_indices,
            "target_sample_annotation_tokens": target_annotations,
        })

    if str(frame_summaries[0]["sample_token"]) != args.expect_f0_sample_token:
        parser.error(
            "frame-0 sample token mismatch: got %s expected %s"
            % (frame_summaries[0]["sample_token"], args.expect_f0_sample_token)
        )
    if not all_target:
        parser.error("target instance is missing in at least one frame")

    labels_dir = Path(args.output_dir) / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    window_records = [
        records[front + step * args.hz_factor] for step in range(args.frame_num)
    ]
    with (labels_dir / "data.pkl").open("wb") as handle:
        pickle.dump(window_records, handle)
    summary = {
        "schema_version": "driveloop_candidate_identity_probe.v1",
        "source": "converted_record_fields",
        "candidate": args.candidate_id,
        "frame_count": args.frame_num,
        "raw_root": args.raw_root,
        "target_raw_instance_token": args.instance_token,
        "audit_only": True,
        "all_frames_have_instance_tokens": all_instance,
        "all_frames_have_sample_annotation_tokens": all_annotation,
        "all_frames_have_target": all_target,
        "output_label_path": str(labels_dir / "data.pkl"),
        "frame_summaries": frame_summaries,
    }
    (labels_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({
        "candidate": args.candidate_id,
        "summary": str(labels_dir / "summary.json"),
        "all_frames_have_target": all_target,
        "f0_sample_token": frame_summaries[0]["sample_token"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
