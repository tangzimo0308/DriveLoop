from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from dreamer_datasets import load_dataset

from scripts.run_dd2_structural_audit import load_config, make_transform, tensor_sig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU-only audit for per-frame DriveLoop boxes3d overrides reaching DD2 transform tensors."
    )
    parser.add_argument(
        "--config-path",
        default="dreamer-train/projects/DriveDreamer2/configs/drivedreamer2_img_cond_mini_local.py",
    )
    parser.add_argument(
        "--output",
        default="outputs/driveloop/per_frame_boxes3d_transform_surface_audit/latest.json",
    )
    parser.add_argument("--frame-indices", default="0,1,2,3")
    parser.add_argument("--target-frame-indices", default="0,2")
    parser.add_argument("--box-dx", type=float, default=0.75)
    parser.add_argument("--require-change", action="store_true")
    return parser.parse_args()


def _parse_frame_indices(value: str) -> list[int]:
    frames: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        frames.append(int(part))
    if not frames:
        raise ValueError("at least one frame index is required")
    return frames


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _shifted_existing_box(raw_dataset: Any, source_index: int, dx: float) -> list[float]:
    sample = raw_dataset[source_index]
    boxes = np.asarray(sample["boxes3d"], dtype=np.float32)
    if boxes.size == 0:
        raise RuntimeError(f"sample {source_index} has no boxes3d")
    box = boxes[0].copy()
    box[0] += dx
    if box.shape[0] > 1:
        box[1] += dx * 0.2
    return [round(float(value), 6) for value in box.tolist()]


def _build_override_spec(raw_dataset: Any, target_frame_indices: Iterable[int], box_dx: float) -> dict[str, Any]:
    entries = []
    for offset, frame_idx in enumerate(target_frame_indices):
        direction = box_dx if offset % 2 == 0 else -box_dx
        entries.append(
            {
                "frame_idx": int(frame_idx),
                "category": "car",
                "box3d": _shifted_existing_box(raw_dataset, int(frame_idx), direction),
                "source": "per_frame_transform_surface_audit_shifted_existing_box",
                "provenance": "audit_only_no_semantic_success_claim",
            }
        )

    return {
        "schema_version": "driveloop_dd2_override.v0",
        "boxes3d": {"per_frame_append": entries},
        "image_hdmap": {
            "mode": "keep_baseline",
            "reason": "per_frame_boxes3d_transform_surface_audit_only",
        },
    }


def _collect_rows(label: str, data_cfg: dict[str, Any], frame_indices: list[int], override_json: dict[str, Any] | None, audit_path: Path) -> list[dict[str, Any]]:
    if override_json is None:
        os.environ.pop("DRIVELOOP_DD2_OVERRIDE_JSON", None)
        os.environ.pop("DRIVELOOP_DD2_OVERRIDE_AUDIT_PATH", None)
    else:
        os.environ["DRIVELOOP_DD2_OVERRIDE_JSON"] = json.dumps(override_json, sort_keys=True)
        os.environ["DRIVELOOP_DD2_OVERRIDE_AUDIT_PATH"] = str(audit_path)

    dataset = load_dataset(data_cfg["data_or_config"])
    dataset.set_transform(make_transform(data_cfg["transform"]))

    rows = []
    for index in frame_indices:
        sample = dataset[index]
        box_sig = tensor_sig(sample["box_downsampler_input"])
        rows.append(
            {
                "label": label,
                "index": int(index),
                "frame_idx": _jsonable(sample.get("frame_idx")),
                "box_downsampler_input": box_sig,
                "box_sha256": box_sig["sha256"],
                "box_sum": box_sig["sum"],
            }
        )
    return rows


def _compare_rows(baseline_rows: list[dict[str, Any]], override_rows: list[dict[str, Any]], target_frame_indices: Iterable[int]) -> list[dict[str, Any]]:
    target_set = {int(value) for value in target_frame_indices}
    comparisons = []
    for before, after in zip(baseline_rows, override_rows):
        frame_idx = int(after["frame_idx"])
        comparisons.append(
            {
                "index": after["index"],
                "frame_idx": frame_idx,
                "targeted_frame": frame_idx in target_set,
                "box_downsampler_changed": before["box_sha256"] != after["box_sha256"],
                "baseline_box_sha256": before["box_sha256"],
                "override_box_sha256": after["box_sha256"],
            }
        )
    return comparisons


def _read_audit_entries(audit_path: Path) -> list[dict[str, Any]]:
    if not audit_path.exists():
        return []
    return [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _build_checks(
    comparisons: list[dict[str, Any]],
    audit_entries: list[dict[str, Any]],
    frame_indices: Iterable[int],
    target_frame_indices: Iterable[int],
) -> dict[str, Any]:
    frame_list = [int(value) for value in frame_indices]
    target_set = {int(value) for value in target_frame_indices}
    targeted = [item for item in comparisons if item["targeted_frame"]]
    non_targeted = [item for item in comparisons if not item["targeted_frame"]]

    audit_by_frame = {
        int(entry.get("sample_identity", {}).get("frame_idx")): entry
        for entry in audit_entries
        if entry.get("sample_identity", {}).get("frame_idx") is not None
    }

    target_audit_image_box_changed = all(
        audit_by_frame.get(frame, {}).get("changed", {}).get("image_box") is True
        for frame in target_set
    )
    non_target_audit_skipped = all(
        any(
            item.get("mode") == "per_frame_append" and item.get("reason") == "no_matching_frame_idx"
            for item in audit_by_frame.get(frame, {}).get("skipped", [])
        )
        for frame in frame_list
        if frame not in target_set
    )

    checks = {
        "targeted_frames_changed": bool(targeted) and all(item["box_downsampler_changed"] for item in targeted),
        "non_targeted_frames_unchanged": all(not item["box_downsampler_changed"] for item in non_targeted),
        "override_audit_entry_count_matches": len(audit_entries) == len(frame_list),
        "target_audit_image_box_changed": target_audit_image_box_changed,
        "non_target_audit_skipped": non_target_audit_skipped,
    }
    checks["structural_condition_surface_verified"] = all(checks.values())
    return checks


def main() -> None:
    args = parse_args()
    frame_indices = _parse_frame_indices(args.frame_indices)
    target_frame_indices = _parse_frame_indices(args.target_frame_indices)

    config = load_config(Path(args.config_path))
    data_cfg = config["dataloaders"]["test"]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path = output_path.with_suffix(".override_audit.jsonl")
    if audit_path.exists():
        audit_path.unlink()

    try:
        raw_dataset = load_dataset(data_cfg["data_or_config"])
        override_json = _build_override_spec(raw_dataset, target_frame_indices, args.box_dx)

        baseline_rows = _collect_rows("baseline", data_cfg, frame_indices, None, audit_path)
        override_rows = _collect_rows("override", data_cfg, frame_indices, override_json, audit_path)
        comparisons = _compare_rows(baseline_rows, override_rows, target_frame_indices)
        audit_entries = _read_audit_entries(audit_path)
        checks = _build_checks(comparisons, audit_entries, frame_indices, target_frame_indices)

        report = {
            "schema_version": "driveloop_per_frame_boxes3d_transform_surface_audit.v0",
            "scope": {
                "gpu_generation_run": False,
                "transform_only": True,
                "claim_boundary": [
                    "per-frame boxes3d/image_box structural conditioning only",
                    "no actor trajectory control claim",
                    "no velocity-conditioned generation claim",
                    "no lane-change semantic success claim",
                    "no GPU video success claim",
                ],
            },
            "config_path": args.config_path,
            "frame_indices": frame_indices,
            "target_frame_indices": target_frame_indices,
            "override_json": override_json,
            "baseline_rows": baseline_rows,
            "override_rows": override_rows,
            "comparisons": comparisons,
            "override_audit_path": str(audit_path),
            "override_audit_entry_count": len(audit_entries),
            "override_audit_preview": audit_entries[:4],
            "checks": checks,
        }
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

        print(json.dumps(
            {
                "output": str(output_path),
                "override_audit_path": str(audit_path),
                "checks": checks,
                "comparisons": comparisons,
            },
            indent=2,
            sort_keys=True,
        ))

        if args.require_change and not checks["structural_condition_surface_verified"]:
            raise SystemExit("Per-frame boxes3d transform surface verification failed.")
    finally:
        os.environ.pop("DRIVELOOP_DD2_OVERRIDE_JSON", None)
        os.environ.pop("DRIVELOOP_DD2_OVERRIDE_AUDIT_PATH", None)


if __name__ == "__main__":
    main()
