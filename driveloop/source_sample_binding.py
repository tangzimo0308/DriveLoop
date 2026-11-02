from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from scripts.run_dd2_batch_sampler_audit import (
    candidate_camera_starts,
    load_records,
    selected_frame_indices,
)


def _as_set(values: Iterable[Any]) -> set[str]:
    return {str(value) for value in values if value not in (None, "")}


def _identity_sample_tokens(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = data.get("frame_summaries", [])
    if not isinstance(frames, list):
        return set()
    return _as_set(frame.get("sample_token") for frame in frames if isinstance(frame, dict))


def build_source_sample_binding(
    dataset_dir: str | Path,
    *,
    source_candidate_id: str | None = None,
    sample_token: str | None = None,
    scene_token: str | None = None,
    instance_token: str | None = None,
    identity_summary_path: str | Path | None = None,
    frame_num: int = 8,
    hz_factor: int = 3,
    video_split_rate: int = 1,
    multiview: bool = True,
    candidate_offset: int = 0,
) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    labels_path = dataset_dir / "labels" / "data.pkl"
    identity_path = Path(identity_summary_path) if identity_summary_path else None

    selector = {
        "source_candidate_id": source_candidate_id,
        "sample_token": sample_token,
        "scene_token": scene_token,
        "instance_token": instance_token,
        "identity_summary_path": str(identity_path) if identity_path else None,
    }
    identity_sample_tokens = _identity_sample_tokens(identity_path)
    target_sample_tokens = _as_set([sample_token]) | identity_sample_tokens
    target_scene_tokens = _as_set([scene_token])

    requested = bool(source_candidate_id or target_sample_tokens or target_scene_tokens or instance_token)
    if not requested:
        return {
            "schema_version": "driveloop_source_sample_binding.v0",
            "requested": False,
            "ready": False,
            "selector": selector,
            "reason": "no_source_sample_selector_requested",
        }

    if not labels_path.exists():
        return {
            "schema_version": "driveloop_source_sample_binding.v0",
            "requested": True,
            "ready": False,
            "selector": selector,
            "labels_path": str(labels_path),
            "reason": "dd2_labels_data_missing",
        }

    if identity_path is not None and not identity_sample_tokens:
        return {
            "schema_version": "driveloop_source_sample_binding.v1",
            "requested": True, "ready": False, "selector": selector,
            "reason": "source_identity_summary_missing_or_empty",
        }

    records = load_records(labels_path)
    starts = candidate_camera_starts(
        records,
        frame_num=frame_num,
        hz_factor=hz_factor,
        video_split_rate=video_split_rate,
        multiview=multiview,
    )

    matches = []
    for candidate_index, camera_starts in enumerate(starts):
        selected_indices = selected_frame_indices(
            camera_starts, frame_num=frame_num, hz_factor=hz_factor,
        )
        if any(index < 0 or index >= len(records) for index in selected_indices):
            continue
        selected_records = [records[index] for index in selected_indices]
        selected_sample_tokens = _as_set(record.get("sample_token") for record in selected_records)
        selected_scene_tokens = _as_set(record.get("scene_token") for record in selected_records)
        if not (target_sample_tokens or target_scene_tokens or instance_token):
            continue  # A display candidate ID alone is not a dataset-stable identity.
        if target_sample_tokens and not target_sample_tokens.issubset(selected_sample_tokens):
            continue
        if target_scene_tokens and selected_scene_tokens != target_scene_tokens:
            continue
        front_record_index = camera_starts[1] if multiview and len(camera_starts) > 1 else camera_starts[0]
        front_records = [records[front_record_index + step * hz_factor] for step in range(frame_num)]
        if instance_token:
            # An explicitly requested actor must be identified in every selected
            # front frame. Category matches cannot substitute for actor identity.
            covered = []
            for record in front_records:
                tokens = record.get("instance_tokens")
                labels = record.get("ori_labels3d", [])
                boxes = record.get("boxes3d")
                if tokens is None or boxes is None or len(tokens) != len(labels) or len(tokens) != len(boxes):
                    covered.append(False)
                else:
                    covered.append(str(instance_token) in _as_set(tokens))
            if not all(covered):
                continue
        matches.append({
            "schema_version": "driveloop_source_sample_binding.v1",
            "requested": True, "ready": True, "selector": selector,
            "dataset_dir": str(dataset_dir), "labels_path": str(labels_path),
            "record_count": len(records), "candidate_start_count": len(starts),
            "dd2_batch_skip": candidate_index,
            "rebinding": None,
            "front_record_index": front_record_index,
            "front_record": {
                key: records[front_record_index].get(key)
                for key in ("sample_token", "scene_token", "cam_type", "frame_idx", "scene_description")
            },
            "selected_frame_indices_preview": selected_indices[:24],
            "matched_sample_tokens": sorted(target_sample_tokens & selected_sample_tokens),
            "matched_scene_tokens": sorted(target_scene_tokens & selected_scene_tokens),
            "unique_selected_sample_token_count": len(selected_sample_tokens),
            "unique_selected_scene_token_count": len(selected_scene_tokens),
            "instance_coverage": ({"matched_front_frames": frame_num, "required_front_frames": frame_num}
                                  if instance_token else None),
            "claim_boundary": {
                "runtime_sample_selector_verified": True,
                "source_sample_binding_is_not_gpu_approval": True,
                "source_sample_binding_is_not_video_semantic_success": True,
            },
        })

    if matches:
        failure_reason = None
        if len(matches) != 1:
            failure_reason = "ambiguous_source_window_requires_sample_identity"
        elif candidate_offset:
            shifted = (matches[0]["dd2_batch_skip"] + int(candidate_offset)) % len(starts)
            if shifted != matches[0]["dd2_batch_skip"]:
                failure_reason = "source_offset_would_violate_identity"
            else:
                matches[0]["rebinding"] = {"candidate_offset": int(candidate_offset), "effective_noop": True}
        if failure_reason is None:
            return matches[0]
        return {
            "schema_version": "driveloop_source_sample_binding.v1",
            "requested": True, "ready": False, "selector": selector,
            "dataset_dir": str(dataset_dir), "labels_path": str(labels_path),
            "record_count": len(records), "candidate_start_count": len(starts),
            "matching_candidate_indices": [match["dd2_batch_skip"] for match in matches],
            "reason": failure_reason,
        }

    return {
        "schema_version": "driveloop_source_sample_binding.v0",
        "requested": True,
        "ready": False,
        "selector": selector,
        "dataset_dir": str(dataset_dir),
        "labels_path": str(labels_path),
        "record_count": len(records),
        "candidate_start_count": len(starts),
        "target_sample_token_count": len(target_sample_tokens),
        "target_scene_token_count": len(target_scene_tokens),
        "reason": ("no_dd2_candidate_covers_requested_instance" if instance_token
                   else "no_dd2_candidate_contains_requested_source_tokens"),
        "claim_boundary": {
            "runtime_sample_selector_verified": False,
            "source_sample_binding_is_not_gpu_approval": True,
            "source_sample_binding_is_not_video_semantic_success": True,
        },
    }
