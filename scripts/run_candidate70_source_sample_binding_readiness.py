#!/usr/bin/env python3
"""Build a non-GPU readiness gate for candidate70 source-sample binding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from driveloop.source_sample_binding import build_source_sample_binding


DEFAULT_IDENTITY_SUMMARY = Path(
    "outputs/driveloop/candidate70_converter_identity_probe/cam_front_8/v0.0.1/labels/summary.json"
)
DEFAULT_FAILED_ALIGNMENT = Path(
    "outputs/driveloop/prompt_video_alignment_eval/"
    "candidate70_night_cut_in_gpu_smoke_manual_review/"
    "prompt_video_alignment_evaluation.json"
)
DEFAULT_RUNNER = Path("scripts/run_driveloop_drivedreamer2.py")
DEFAULT_BACKEND = Path("driveloop/backends/drivedreamer2.py")
DEFAULT_RUNTIME_DATASET_DIR = Path(
    "/mnt/driveloop_full/processed/nuscenes/v1.0-trainval/candidate70_source_bound/cam_all_train/v0.0.1"
)
DEFAULT_OUTPUT = Path(
    "outputs/driveloop/source_sample_binding_readiness/"
    "candidate70_source_sample_binding_readiness.json"
)


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def has_runtime_sample_selector(text: str) -> bool:
    selector_terms = [
        "--source-candidate-id",
        "--scene-token",
        "--sample-token",
        "--instance-token",
        "--source-identity-summary",
        "DRIVELOOP_DD2_SAMPLE_TOKEN",
        "DRIVELOOP_DD2_SCENE_TOKEN",
        "source_candidate_id",
        "sample_token_selector",
        "scene_token_selector",
        "source_sample_binding",
        "build_source_sample_binding",
    ]
    return any(term in text for term in selector_terms)



def inspect_runtime_generation_dataset(dataset_dir: Path) -> Dict[str, Any]:
    required_paths = {
        "dataset_config": "config.json",
        "labels_config": "labels/config.json",
        "labels_data": "labels/data.pkl",
        "images_config": "images/config.json",
        "images_lmdb": "images/data.mdb",
        "hdmaps_config": "hdmaps/config.json",
        "hdmaps_lmdb": "hdmaps/data.mdb",
    }
    path_status = {}
    missing = []
    for name, relative_path in required_paths.items():
        path = dataset_dir / relative_path
        exists = path.exists()
        path_status[name] = {
            "relative_path": relative_path,
            "path": str(path),
            "exists": exists,
        }
        if not exists:
            missing.append(name)

    return {
        "dataset_dir": str(dataset_dir),
        "complete": not missing,
        "required_paths": path_status,
        "missing_required_paths": missing,
        "claim_boundary": {
            "generation_dataset_complete_is_not_gpu_approval": True,
            "generation_dataset_complete_is_not_video_semantic_success": True,
        },
    }


def build_gate(
    identity_summary_path: Path = DEFAULT_IDENTITY_SUMMARY,
    failed_alignment_path: Path = DEFAULT_FAILED_ALIGNMENT,
    runner_path: Path = DEFAULT_RUNNER,
    backend_path: Path = DEFAULT_BACKEND,
    runtime_dataset_dir: Path = DEFAULT_RUNTIME_DATASET_DIR,
    source_selector_frame_num: int = 8,
    source_selector_hz_factor: int = 3,
    source_selector_video_split_rate: int = 1,
    source_selector_multiview: bool = True,
) -> Dict[str, Any]:
    identity = load_json(identity_summary_path)
    failed_alignment = load_json(failed_alignment_path)
    runner_text = read_text(runner_path)
    backend_text = read_text(backend_path)

    frames = identity.get("frame_summaries", [])
    sample_tokens = [
        frame.get("sample_token")
        for frame in frames
        if isinstance(frame, dict) and frame.get("sample_token")
    ]
    cam_tokens = [
        frame.get("cam_token")
        for frame in frames
        if isinstance(frame, dict) and frame.get("cam_token")
    ]

    runner_has_selector = has_runtime_sample_selector(runner_text)
    backend_has_selector = has_runtime_sample_selector(backend_text)

    source_sample_binding = build_source_sample_binding(
        runtime_dataset_dir,
        source_candidate_id=identity.get("candidate") or "candidate70",
        identity_summary_path=identity_summary_path if identity_summary_path.exists() else None,
        instance_token=identity.get("target_raw_instance_token"),
        frame_num=source_selector_frame_num,
        hz_factor=source_selector_hz_factor,
        video_split_rate=source_selector_video_split_rate,
        multiview=source_selector_multiview,
    )
    binding_ready = source_sample_binding.get("ready") is True
    runtime_generation_dataset = inspect_runtime_generation_dataset(runtime_dataset_dir)
    generation_complete = runtime_generation_dataset["complete"]

    checks = {
        "identity_summary_exists": identity_summary_path.exists(),
        "candidate_is_candidate70": identity.get("candidate") == "candidate70",
        "converter_identity_subset_created": identity.get("claim", {}).get(
            "candidate70_converter_derived_identity_subset_created"
        )
        is True,
        "all_frames_have_target": identity.get("all_frames_have_target") is True,
        "all_frames_have_instance_tokens": identity.get("all_frames_have_instance_tokens") is True,
        "all_frames_have_sample_annotation_tokens": identity.get(
            "all_frames_have_sample_annotation_tokens"
        )
        is True,
        "sample_tokens_available": len(sample_tokens) == identity.get("frame_count"),
        "target_raw_instance_token_available": bool(identity.get("target_raw_instance_token")),
        "runner_has_runtime_sample_selector": runner_has_selector,
        "backend_has_runtime_sample_selector": backend_has_selector,
        "runtime_source_sample_binding_ready": binding_ready,
        "runtime_selected_dd2_batch_skip_available": isinstance(
            source_sample_binding.get("dd2_batch_skip"),
            int,
        ),
        "runtime_generation_dataset_complete": generation_complete,
        "failed_alignment_is_measured_failed": failed_alignment.get("interpretation", {}).get(
            "video_semantic_claim"
        )
        == "measured_failed",
    }

    blockers: List[str] = []
    if not checks["identity_summary_exists"]:
        blockers.append("candidate70_identity_summary_missing")
    if not checks["sample_tokens_available"]:
        blockers.append("candidate70_sample_tokens_missing_or_incomplete")
    if not checks["runner_has_runtime_sample_selector"]:
        blockers.append("runner_has_no_candidate70_source_sample_selector")
    if not checks["backend_has_runtime_sample_selector"]:
        blockers.append("backend_has_no_verified_runtime_sample_selector")
    if not checks["runtime_source_sample_binding_ready"]:
        blockers.append("candidate70_source_tokens_not_resolved_to_dd2_runtime_candidate")
    if not (
        checks["runner_has_runtime_sample_selector"]
        and checks["backend_has_runtime_sample_selector"]
        and checks["runtime_source_sample_binding_ready"]
    ):
        blockers.append("blocked_no_verified_runtime_sample_selector")

    generation_blockers: List[str] = []
    if not generation_complete:
        generation_blockers.append("runtime_generation_dataset_incomplete")
        generation_blockers.extend(
            f"missing_runtime_generation_path:{name}"
            for name in runtime_generation_dataset["missing_required_paths"]
        )

    readiness_status = "ready" if not blockers else "blocked_no_verified_runtime_sample_selector"
    runtime_generation_readiness_status = (
        "ready" if generation_complete else "blocked_incomplete_runtime_generation_dataset"
    )
    gpu_blockers = ["source_sample_binding_gate_is_not_gpu_approval"]
    gpu_blockers.extend(blockers)
    gpu_blockers.extend(generation_blockers)

    return {
        "schema_version": "driveloop_candidate70_source_sample_binding_readiness.v1",
        "candidate": "candidate70",
        "readiness_status": readiness_status,
        "source_sample_binding_readiness_status": readiness_status,
        "runtime_generation_readiness_status": runtime_generation_readiness_status,
        "runtime_generation_ready": generation_complete,
        "resolved_dd2_batch_skip": source_sample_binding.get("dd2_batch_skip"),
        "runtime_dataset_dir": str(runtime_dataset_dir),
        "gpu_smoke_allowed": False,
        "does_not_run_gpu": True,
        "does_not_generate_video": True,
        "checks": checks,
        "blockers": list(dict.fromkeys(blockers)),
        "generation_blockers": list(dict.fromkeys(generation_blockers)),
        "gpu_blockers": list(dict.fromkeys(gpu_blockers)),
        "candidate70_source_evidence": {
            "source": identity.get("source"),
            "raw_root": identity.get("raw_root"),
            "output_label_path": identity.get("output_label_path"),
            "target_raw_instance_token": identity.get("target_raw_instance_token"),
            "frame_count": identity.get("frame_count"),
            "first_sample_token": sample_tokens[0] if sample_tokens else None,
            "last_sample_token": sample_tokens[-1] if sample_tokens else None,
            "unique_sample_token_count": len(set(sample_tokens)),
            "unique_cam_token_count": len(set(cam_tokens)),
        },
        "runtime_binding_assessment": {
            "runner_path": str(runner_path),
            "backend_path": str(backend_path),
            "runtime_dataset_dir": str(runtime_dataset_dir),
            "runtime_sample_selector_code_present": runner_has_selector and backend_has_selector,
            "runtime_sample_selector_resolved": binding_ready,
            "runtime_sample_selector_verified": runner_has_selector and backend_has_selector and binding_ready,
            "resolved_dd2_batch_skip": source_sample_binding.get("dd2_batch_skip"),
            "source_sample_binding": source_sample_binding,
            "runtime_generation_dataset_complete": generation_complete,
            "runtime_generation_dataset": runtime_generation_dataset,
            "current_failure_interpretation": (
                "candidate70 source binding is assessed separately from DD2 generation dataset "
                "completeness; GPU remains blocked unless source tokens resolve and the DD2 "
                "runtime dataset contains required generation artifacts."
            ),
        },
        "claim_boundary": {
            "source_sample_binding_gate_is_not_gpu_approval": True,
            "runtime_generation_dataset_ready_is_required_before_gpu": True,
            "runtime_generation_dataset_complete_is_not_video_semantic_success": True,
            "converter_identity_subset_is_not_runtime_binding": True,
            "sample_tokens_available_is_not_video_semantic_success": True,
            "measured_failed_video_is_not_semantic_success": True,
            "semantic_success_claim_allowed": False,
        },
        "next_required_steps": [
            "use full trainval runtime dataset for candidate70 source-sample binding",
            "prove candidate70 source sample is selected by runtime, not only converter/audit outputs",
            "complete DD2 runtime generation dataset config/images/hdmaps before GPU",
            "keep GPU blocked until source binding and generation dataset readiness are separately approved",
        ],
        "sources": {
            "identity_summary": {"path": str(identity_summary_path), "exists": identity_summary_path.exists()},
            "failed_alignment": {"path": str(failed_alignment_path), "exists": failed_alignment_path.exists()},
            "runner": {"path": str(runner_path), "exists": runner_path.exists()},
            "backend": {"path": str(backend_path), "exists": backend_path.exists()},
            "runtime_dataset_dir": {"path": str(runtime_dataset_dir), "exists": runtime_dataset_dir.exists()},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build candidate70 source-sample binding readiness gate."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runtime-dataset-dir", type=Path, default=DEFAULT_RUNTIME_DATASET_DIR)
    parser.add_argument("--source-selector-frame-num", type=int, default=8)
    parser.add_argument("--source-selector-hz-factor", type=int, default=3)
    parser.add_argument("--source-selector-video-split-rate", type=int, default=1)
    parser.add_argument("--source-selector-single-view", action="store_true")
    args = parser.parse_args()

    gate = build_gate(
        runtime_dataset_dir=args.runtime_dataset_dir,
        source_selector_frame_num=args.source_selector_frame_num,
        source_selector_hz_factor=args.source_selector_hz_factor,
        source_selector_video_split_rate=args.source_selector_video_split_rate,
        source_selector_multiview=not args.source_selector_single_view,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)
    print(json.dumps(gate, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
