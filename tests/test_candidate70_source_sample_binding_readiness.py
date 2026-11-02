import json
import pickle
from pathlib import Path

from scripts.run_candidate70_source_sample_binding_readiness import build_gate


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def write_runtime_dataset(path: Path, sample_tokens=("sample_a", "sample_b")) -> Path:
    labels = path / "labels"
    labels.mkdir(parents=True)
    records = []
    for index, token in enumerate(sample_tokens):
        records.append(
            {
                "cam_type": "CAM_FRONT",
                "frame_idx": index,
                "video_length": len(sample_tokens),
                "sample_token": token,
                "scene_token": "scene_a",
                "scene_description": "candidate70 target sequence",
                "labels3d": [["vehicle", "motorcycle"]],
                "ori_labels3d": ["vehicle.motorcycle"],
                "instance_tokens": ["21cdc9f24c614a6197fd044379697197"],
                "boxes3d": [[0., 1., 20., 0.8, 1.4, 2.2, 0., 0., 0.]],
            }
        )
    with (labels / "data.pkl").open("wb") as handle:
        pickle.dump(records, handle)
    write_json(
        labels / "config.json",
        {
            "_class_name": "PklDataset",
            "_key_names": sorted(records[0].keys()) if records else [],
            "data_size": len(records),
        },
    )
    return path


def write_identity_summary(path: Path, frame_count: int = 2) -> None:
    write_json(
        path,
        {
            "candidate": "candidate70",
            "source": "current_converter_get_cam_label",
            "raw_root": "/data/projects/DriveLoop/data/raw/nuscenes",
            "output_label_path": "outputs/driveloop/candidate70_converter_identity_probe/cam_front_8/v0.0.1/labels/data.pkl",
            "target_raw_instance_token": "21cdc9f24c614a6197fd044379697197",
            "frame_count": frame_count,
            "all_frames_have_target": True,
            "all_frames_have_instance_tokens": True,
            "all_frames_have_sample_annotation_tokens": True,
            "claim": {
                "candidate70_converter_derived_identity_subset_created": True,
            },
            "frame_summaries": [
                {
                    "sample_token": "sample_a",
                    "cam_token": "cam_a",
                    "target_present": True,
                },
                {
                    "sample_token": "sample_b",
                    "cam_token": "cam_b",
                    "target_present": True,
                },
            ][:frame_count],
        },
    )


def write_failed_alignment(path: Path) -> None:
    write_json(
        path,
        {
            "interpretation": {
                "video_semantic_claim": "measured_failed",
            },
        },
    )


def test_candidate70_source_sample_binding_blocks_without_runtime_selector(tmp_path):
    identity_summary = tmp_path / "summary.json"
    failed_alignment = tmp_path / "alignment.json"
    runner = tmp_path / "run_driveloop_drivedreamer2.py"
    backend = tmp_path / "drivedreamer2.py"
    runtime_dataset = write_runtime_dataset(tmp_path / "runtime_dataset")

    write_identity_summary(identity_summary)
    write_failed_alignment(failed_alignment)
    runner.write_text("parser.add_argument('--prompt')\n", encoding="utf-8")
    backend.write_text("baseline_video = self.baseline_output_dir / '000000.mp4'\n", encoding="utf-8")

    gate = build_gate(
        identity_summary_path=identity_summary,
        failed_alignment_path=failed_alignment,
        runner_path=runner,
        backend_path=backend,
        runtime_dataset_dir=runtime_dataset,
        source_selector_frame_num=2,
        source_selector_hz_factor=1,
        source_selector_multiview=False,
    )

    assert gate["readiness_status"] == "blocked_no_verified_runtime_sample_selector"
    assert gate["gpu_smoke_allowed"] is False
    assert gate["does_not_run_gpu"] is True
    assert gate["does_not_generate_video"] is True
    assert gate["checks"]["sample_tokens_available"] is True
    assert gate["checks"]["runner_has_runtime_sample_selector"] is False
    assert gate["checks"]["backend_has_runtime_sample_selector"] is False
    assert gate["checks"]["runtime_source_sample_binding_ready"] is True
    assert gate["checks"]["failed_alignment_is_measured_failed"] is True
    assert "blocked_no_verified_runtime_sample_selector" in gate["blockers"]
    assert gate["candidate70_source_evidence"]["first_sample_token"] == "sample_a"
    assert gate["candidate70_source_evidence"]["last_sample_token"] == "sample_b"
    assert gate["claim_boundary"]["semantic_success_claim_allowed"] is False


def test_candidate70_source_sample_binding_blocks_when_tokens_do_not_resolve(tmp_path):
    identity_summary = tmp_path / "summary.json"
    failed_alignment = tmp_path / "alignment.json"
    runner = tmp_path / "run_driveloop_drivedreamer2.py"
    backend = tmp_path / "drivedreamer2.py"
    runtime_dataset = write_runtime_dataset(tmp_path / "runtime_dataset", sample_tokens=("other_a", "other_b"))

    write_identity_summary(identity_summary)
    write_failed_alignment(failed_alignment)
    runner.write_text("parser.add_argument('--sample-token')\n", encoding="utf-8")
    backend.write_text("source_sample_binding = build_source_sample_binding(...)\n", encoding="utf-8")

    gate = build_gate(
        identity_summary_path=identity_summary,
        failed_alignment_path=failed_alignment,
        runner_path=runner,
        backend_path=backend,
        runtime_dataset_dir=runtime_dataset,
        source_selector_frame_num=2,
        source_selector_hz_factor=1,
        source_selector_multiview=False,
    )

    assert gate["readiness_status"] == "blocked_no_verified_runtime_sample_selector"
    assert gate["checks"]["runner_has_runtime_sample_selector"] is True
    assert gate["checks"]["backend_has_runtime_sample_selector"] is True
    assert gate["checks"]["runtime_source_sample_binding_ready"] is False
    assert "candidate70_source_tokens_not_resolved_to_dd2_runtime_candidate" in gate["blockers"]
    assert gate["runtime_binding_assessment"]["runtime_sample_selector_verified"] is False


def test_candidate70_source_sample_binding_ready_only_when_runtime_selector_resolves(tmp_path):
    identity_summary = tmp_path / "summary.json"
    failed_alignment = tmp_path / "alignment.json"
    runner = tmp_path / "run_driveloop_drivedreamer2.py"
    backend = tmp_path / "drivedreamer2.py"
    runtime_dataset = write_runtime_dataset(tmp_path / "runtime_dataset")

    write_identity_summary(identity_summary)
    write_failed_alignment(failed_alignment)
    runner.write_text("parser.add_argument('--sample-token')\n", encoding="utf-8")
    backend.write_text("source_sample_binding = build_source_sample_binding(...)\n", encoding="utf-8")

    gate = build_gate(
        identity_summary_path=identity_summary,
        failed_alignment_path=failed_alignment,
        runner_path=runner,
        backend_path=backend,
        runtime_dataset_dir=runtime_dataset,
        source_selector_frame_num=2,
        source_selector_hz_factor=1,
        source_selector_multiview=False,
    )

    assert gate["readiness_status"] == "ready"
    assert gate["runtime_binding_assessment"]["runtime_sample_selector_code_present"] is True
    assert gate["runtime_binding_assessment"]["runtime_sample_selector_resolved"] is True
    assert gate["runtime_binding_assessment"]["runtime_sample_selector_verified"] is True
    assert gate["runtime_binding_assessment"]["resolved_dd2_batch_skip"] == 0
    assert gate["source_sample_binding_readiness_status"] == "ready"
    assert gate["runtime_generation_readiness_status"] == "blocked_incomplete_runtime_generation_dataset"
    assert gate["checks"]["runtime_generation_dataset_complete"] is False
    assert "runtime_generation_dataset_incomplete" in gate["generation_blockers"]
    assert gate["gpu_smoke_allowed"] is False


def write_generation_complete_runtime_dataset(path: Path) -> Path:
    runtime_dataset = write_runtime_dataset(path)
    write_json(
        runtime_dataset / "config.json",
        {
            "_class_name": "Dataset",
            "config_paths": [
                "labels/config.json",
                "images/config.json",
                "hdmaps/config.json",
            ],
        },
    )
    for name in ("images", "hdmaps"):
        data_dir = runtime_dataset / name
        data_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            data_dir / "config.json",
            {
                "_class_name": "LmdbDataset",
                "data_size": 2,
            },
        )
        (data_dir / "data.mdb").write_bytes(b"placeholder")
    return runtime_dataset


def test_candidate70_runtime_generation_dataset_complete_is_reported(tmp_path):
    identity_summary = tmp_path / "summary.json"
    failed_alignment = tmp_path / "alignment.json"
    runner = tmp_path / "run_driveloop_drivedreamer2.py"
    backend = tmp_path / "drivedreamer2.py"
    runtime_dataset = write_generation_complete_runtime_dataset(tmp_path / "runtime_dataset")

    write_identity_summary(identity_summary)
    write_failed_alignment(failed_alignment)
    runner.write_text("parser.add_argument('--sample-token')\n", encoding="utf-8")
    backend.write_text("source_sample_binding = build_source_sample_binding(...)\n", encoding="utf-8")

    gate = build_gate(
        identity_summary_path=identity_summary,
        failed_alignment_path=failed_alignment,
        runner_path=runner,
        backend_path=backend,
        runtime_dataset_dir=runtime_dataset,
        source_selector_frame_num=2,
        source_selector_hz_factor=1,
        source_selector_multiview=False,
    )

    assert gate["source_sample_binding_readiness_status"] == "ready"
    assert gate["runtime_generation_readiness_status"] == "ready"
    assert gate["runtime_generation_ready"] is True
    assert gate["resolved_dd2_batch_skip"] == 0
    assert gate["runtime_dataset_dir"] == str(runtime_dataset)
    assert gate["checks"]["runtime_generation_dataset_complete"] is True
    assert gate["generation_blockers"] == []
    assert gate["runtime_binding_assessment"]["runtime_generation_dataset_complete"] is True
    assert gate["gpu_smoke_allowed"] is False
