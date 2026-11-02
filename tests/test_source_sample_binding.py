from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

from driveloop.backends.drivedreamer2 import DriveDreamer2Backend
from driveloop.schema import DriveLoopRequest
from driveloop.source_sample_binding import build_source_sample_binding


def write_dataset(root: Path) -> Path:
    labels = root / "labels"
    labels.mkdir(parents=True)
    records = [
        {
            "cam_type": "CAM_FRONT",
            "frame_idx": 0,
            "video_length": 4,
            "sample_token": "sample_a",
            "scene_token": "scene_0",
            "scene_description": "target start",
            "labels3d": [["vehicle", "motorcycle"]],
            "ori_labels3d": ["vehicle.motorcycle"],
        },
        {
            "cam_type": "CAM_FRONT",
            "frame_idx": 1,
            "video_length": 4,
            "sample_token": "sample_b",
            "scene_token": "scene_0",
            "scene_description": "target second",
            "labels3d": [["vehicle", "motorcycle"]],
            "ori_labels3d": ["vehicle.motorcycle"],
        },
        {
            "cam_type": "CAM_FRONT",
            "frame_idx": 2,
            "video_length": 4,
            "sample_token": "sample_c",
            "scene_token": "scene_1",
            "scene_description": "other start",
            "labels3d": [["vehicle", "car"]],
            "ori_labels3d": ["vehicle.car"],
        },
        {
            "cam_type": "CAM_FRONT",
            "frame_idx": 3,
            "video_length": 4,
            "sample_token": "sample_d",
            "scene_token": "scene_1",
            "scene_description": "other second",
            "labels3d": [["vehicle", "car"]],
            "ori_labels3d": ["vehicle.car"],
        },
    ]
    with (labels / "data.pkl").open("wb") as handle:
        pickle.dump(records, handle)
    return root


def write_identity(path: Path) -> Path:
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "candidate": "candidate70",
                "frame_summaries": [
                    {"sample_token": "sample_a"},
                    {"sample_token": "sample_b"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_source_sample_binding_maps_identity_tokens_to_dd2_batch_skip(tmp_path: Path):
    dataset = write_dataset(tmp_path / "dataset")
    identity = write_identity(tmp_path / "identity" / "summary.json")

    binding = build_source_sample_binding(
        dataset,
        source_candidate_id="candidate70",
        identity_summary_path=identity,
        frame_num=2,
        hz_factor=1,
        video_split_rate=1,
        multiview=False,
    )

    assert binding["ready"] is True
    assert binding["dd2_batch_skip"] == 0
    assert binding["front_record_index"] == 0
    assert binding["matched_sample_tokens"] == ["sample_a", "sample_b"]
    assert binding["claim_boundary"]["source_sample_binding_is_not_gpu_approval"] is True


def test_source_sample_binding_blocks_when_tokens_are_not_in_dd2_dataset(tmp_path: Path):
    dataset = write_dataset(tmp_path / "dataset")

    binding = build_source_sample_binding(
        dataset,
        sample_token="missing_sample",
        frame_num=2,
        hz_factor=1,
        video_split_rate=1,
        multiview=False,
    )

    assert binding["ready"] is False
    assert binding["reason"] == "no_dd2_candidate_contains_requested_source_tokens"


def test_drivedreamer2_backend_sets_runtime_batch_skip_from_source_binding(monkeypatch, tmp_path: Path):
    dataset = write_dataset(tmp_path / "dataset")
    identity = write_identity(tmp_path / "identity" / "summary.json")
    baseline_output_dir = tmp_path / "baseline"
    baseline_output_dir.mkdir()
    baseline_video = baseline_output_dir / "000000.mp4"
    captured = {}

    def fake_run(cmd, cwd, env, check, text, timeout):
        captured["env"] = env
        baseline_video.write_bytes(b"fake video")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("driveloop.backends.drivedreamer2.subprocess.run", fake_run)

    backend = DriveDreamer2Backend(
        project_root=tmp_path,
        baseline_output_dir=baseline_output_dir,
        baseline_dataset_dir=dataset,
        artifact_dir=tmp_path / "artifacts",
        source_candidate_id="candidate70",
        source_identity_summary_path=identity,
        source_selector_frame_num=2,
        source_selector_hz_factor=1,
        source_selector_video_split_rate=1,
        source_selector_multiview=False,
    )

    generation = backend.generate(DriveLoopRequest(prompt="target prompt"), iteration=0)

    assert captured["env"]["DRIVELOOP_DD2_BATCH_SKIP"] == "0"
    assert generation.metadata["dd2_source_sample_binding"]["ready"] is True
    assert generation.metadata["dd2_source_sample_binding"]["dd2_batch_skip"] == 0
    assert generation.metadata["dd2_baseline_structural_snapshot"]["sample"]["sample_token"] == "sample_a"


def test_source_sample_binding_blocks_partial_identity_token_match(tmp_path: Path):
    dataset = write_dataset(tmp_path / "dataset")
    identity = write_identity(tmp_path / "identity" / "summary.json")
    data = json.loads(identity.read_text(encoding="utf-8"))
    data["frame_summaries"].append({"sample_token": "missing_sample"})
    identity.write_text(json.dumps(data), encoding="utf-8")

    binding = build_source_sample_binding(
        dataset,
        source_candidate_id="candidate70",
        identity_summary_path=identity,
        frame_num=2,
        hz_factor=1,
        video_split_rate=1,
        multiview=False,
    )

    assert binding["ready"] is False
    assert binding["reason"] == "no_dd2_candidate_contains_requested_source_tokens"


def actor_dataset(root, *, present=(False, False, True, True), tokens_available=True):
    dataset = write_dataset(root)
    path = dataset / 'labels' / 'data.pkl'
    with path.open('rb') as handle:
        records = pickle.load(handle)
    for i, record in enumerate(records):
        record['scene_token'] = 'scene_0'
        record['boxes3d'] = [[0., 1., 20., 2., 2., 4., 0., 0., 0.]]
        if tokens_available:
            record['instance_tokens'] = ['target' if present[i] else 'other']
    with path.open('wb') as handle:
        pickle.dump(records, handle)
    return dataset


def bind_actor(dataset, **kwargs):
    return build_source_sample_binding(dataset, scene_token='scene_0', instance_token='target',
        frame_num=2, hz_factor=1, multiview=False, **kwargs)


def test_binding_skips_same_scene_window_without_requested_actor(tmp_path):
    binding = bind_actor(actor_dataset(tmp_path / 'dataset'))
    assert binding['ready'] is True
    assert binding['dd2_batch_skip'] == 1
    assert binding['instance_coverage']['matched_front_frames'] == 2


def test_binding_rejects_only_partial_actor_coverage(tmp_path):
    binding = bind_actor(actor_dataset(tmp_path / 'dataset', present=(False, True, True, False)))
    assert binding['ready'] is False
    assert binding['reason'] == 'no_dd2_candidate_covers_requested_instance'


def test_binding_requires_actor_identity_annotations(tmp_path):
    binding = bind_actor(actor_dataset(tmp_path / 'dataset', tokens_available=False))
    assert binding['ready'] is False


def test_ambiguous_actor_windows_require_sample_identity(tmp_path):
    dataset = actor_dataset(tmp_path / 'dataset', present=(True, True, True, True))
    binding = bind_actor(dataset)
    assert binding['ready'] is False
    assert binding['matching_candidate_indices'] == [0, 1]
    assert binding['reason'] == 'ambiguous_source_window_requires_sample_identity'
    pinned = bind_actor(dataset, sample_token='sample_c')
    assert pinned['ready'] is True
    assert pinned['dd2_batch_skip'] == 1


def test_missing_identity_file_does_not_fall_back_to_scene(tmp_path):
    binding = bind_actor(actor_dataset(tmp_path / 'dataset'), identity_summary_path=tmp_path / 'missing.json')
    assert binding['ready'] is False
    assert binding['reason'] == 'source_identity_summary_missing_or_empty'


def test_rebinding_offset_cannot_escape_pinned_actor_window(tmp_path):
    binding = bind_actor(actor_dataset(tmp_path / 'dataset'), sample_token='sample_c', candidate_offset=1)
    assert binding['ready'] is False
    assert binding['reason'] == 'source_offset_would_violate_identity'


def test_source_candidate_name_alone_cannot_select_arbitrary_window(tmp_path):
    binding = build_source_sample_binding(actor_dataset(tmp_path / 'dataset'),
        source_candidate_id='candidate1', frame_num=2, hz_factor=1, multiview=False)
    assert binding['ready'] is False
