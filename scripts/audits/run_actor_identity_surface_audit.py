from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any


DEFAULT_CONVERTER = Path("dreamer-datasets/dd_scripts/converters/nuscenes_converter.py")
DEFAULT_VAL_LABELS = Path("/data/projects/DriveLoop/data/processed/nuscenes/v1.0-mini/cam_all_val/v0.0.2/labels/data.pkl")
DEFAULT_TRAIN_LABELS = Path("/data/projects/DriveLoop/data/processed/nuscenes/v1.0-mini/cam_all_train/v0.0.2/labels/data.pkl")
DEFAULT_OUTPUT = Path("outputs/driveloop/actor_identity_surface_audit/mini_actor_identity_surface_audit.json")

ACTOR_IDENTITY_FIELDS = [
    "instance_token",
    "instance_tokens",
    "track_id",
    "track_ids",
    "sample_annotation_token",
    "sample_annotation_tokens",
    "annotation_token",
    "annotation_tokens",
    "ann_token",
    "ann_tokens",
]


def load_first_record(path: Path) -> dict[str, Any]:
    try:
        data = pickle.loads(path.read_bytes())
    except Exception:
        return {}
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    if isinstance(data, dict):
        return data
    return {}


def token_like_keys(record: dict[str, Any]) -> list[str]:
    return sorted(
        key
        for key in record
        if "token" in str(key).lower() or "track" in str(key).lower() or "instance" in str(key).lower()
    )


def shape_or_len(value: Any) -> dict[str, Any]:
    shape = getattr(value, "shape", None)
    try:
        length = len(value)
    except Exception:
        length = None
    return {
        "type": type(value).__name__,
        "shape": list(shape) if shape is not None else None,
        "len": length,
    }


def inspect_label_file(path: Path) -> dict[str, Any]:
    record = load_first_record(path)
    identity_fields = [field for field in ACTOR_IDENTITY_FIELDS if field in record]
    key_summaries = {
        key: shape_or_len(record[key])
        for key in ["boxes3d", "velocities", "labels3d", "ori_labels3d", "attributes"]
        if key in record
    }
    return {
        "path": str(path),
        "exists": path.exists(),
        "record_available": bool(record),
        "key_count": len(record),
        "token_like_keys": token_like_keys(record),
        "actor_identity_fields_present": identity_fields,
        "actor_identity_available": bool(identity_fields),
        "key_summaries": key_summaries,
    }


def inspect_converter(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        text = ""
    return {
        "path": str(path),
        "exists": path.exists(),
        "cam_box_token_observed": "cam_box.token" in text,
        "sample_annotation_lookup_observed": "sample_annotation" in text and "cam_box.token" in text,
        "velocity_uses_annotation_token": "box_velocity(cam_box.token)" in text,
        "converter_has_actor_identity_token_source": "cam_box.token" in text or "sample_annotation" in text,
        "processed_label_identity_write_observed": any(
            pattern in text
            for pattern in [
                "\"instance_token\":",
                "\"instance_tokens\":",
                "\"track_id\":",
                "\"track_ids\":",
                "\"sample_annotation_token\":",
                "\"sample_annotation_tokens\":",
                "\'instance_token\':",
                "\'instance_tokens\':",
                "\'track_id\':",
                "\'track_ids\':",
                "\'sample_annotation_token\':",
                "\'sample_annotation_tokens\':",
            ]
        ),
    }


def build_report(converter: Path, label_paths: list[Path]) -> dict[str, Any]:
    converter_surface = inspect_converter(converter)
    label_surfaces = [inspect_label_file(path) for path in label_paths]
    identity_available = any(item["actor_identity_available"] for item in label_surfaces)

    if identity_available:
        status = "identity_available_in_processed_labels"
    elif converter_surface["cam_box_token_observed"] or converter_surface["sample_annotation_lookup_observed"]:
        status = "identity_available_upstream_but_missing_from_processed_labels"
    else:
        status = "identity_not_observed"

    return {
        "schema_version": "driveloop_actor_identity_surface_audit.v0",
        "status": status,
        "converter_surface": converter_surface,
        "processed_label_surfaces": label_surfaces,
        "claim": {
            "actor_identity_available_in_processed_labels": identity_available,
            "actor_identity_available_upstream": converter_surface["cam_box_token_observed"]
            or converter_surface["sample_annotation_lookup_observed"],
            "runtime_motion_control_connected": False,
            "semantic_success_claim_allowed": False,
        },
        "blockers": [
            "processed_labels_do_not_include_persistent_actor_identity"
        ]
        if not identity_available
        else [],
        "next_actions": [
            "preserve sample_annotation_token or instance_token in converter output labels",
            "rebuild or patch labels before claiming actor identity is available",
            "surface identity through DD2 transform as audit-only metadata before runtime control",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit actor identity surfaces before DD2 runtime control work.")
    parser.add_argument("--converter", type=Path, default=DEFAULT_CONVERTER)
    parser.add_argument("--label", type=Path, action="append", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    label_paths = args.label or [DEFAULT_VAL_LABELS, DEFAULT_TRAIN_LABELS]
    report = build_report(args.converter, label_paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
