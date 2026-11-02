"""C4 end-to-end tests for the ego-frame injection override surface
(boxes3d.per_frame_append_ego): backend emission behind
DRIVELOOP_EGO_INJECTION=1 and per-camera consumption in
driveloop.dd2_override."""
import numpy as np

from driveloop.backends.drivedreamer2 import DriveDreamer2Backend
from driveloop.dd2_override import apply_dd2_override_to_sample

# Real record fixtures (v1.0-mini cam_all_val v0.0.2, first sample_token;
# dumped 2026-07-09, matrices rounded to 6 decimals). Same fixtures as
# tests/test_ego_frame_injection.py.
CAM2EGO_FRONT = [
    [0.010260, 0.008433, 0.999912, 1.722006],
    [-0.999873, 0.012316, 0.010156, 0.004755],
    [-0.012230, -0.999889, 0.008559, 1.494913],
    [0.0, 0.0, 0.0, 1.0],
]
E2G_FRONT = [
    [0.877140, 0.480056, 0.013118, 599.849792],
    [-0.479914, 0.877224, -0.012658, 1647.641113],
    [-0.017584, 0.004807, 0.999834, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]
CAM2EGO_FL = [
    [0.822546, 0.006478, 0.568662, 1.575256],
    [-0.568684, 0.016434, 0.822392, 0.500519],
    [-0.004018, -0.999844, 0.017202, 1.506960],
    [0.0, 0.0, 0.0, 1.0],
]
E2G_FL = [
    [0.877241, 0.479870, 0.013138, 599.791321],
    [-0.479725, 0.877326, -0.012759, 1647.673584],
    [-0.017649, 0.004890, 0.999832, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def _ego_entry(frame_idx=100):
    return {
        "relative_frame_idx": 0,
        "frame_idx": frame_idx,
        "actor_id": "actor_01",
        "synthetic_track_id": "actor_01_synthetic_motion_track",
        "category": "motorcycle",
        # 20 m ahead, 3.5 m LEFT of ego (ego x forward, y left).
        "ego": {"center_ego": [20.0, 3.5, 1.0], "dims": [0.8, 1.4, 2.2], "heading_ego": 0.0},
        "ref_ego2global": E2G_FRONT,
        "sample_identities": [
            {"cam_type": "cam_front", "frame_idx": frame_idx, "sample_token": "tok", "scene_token": "scene"},
            {"cam_type": "cam_front_left", "frame_idx": frame_idx, "sample_token": "tok", "scene_token": "scene"},
        ],
        "motion_surface": "boxes3d.per_frame_append_ego",
    }


def _sample(cam_type, cam2ego, ego2global, frame_idx=100):
    return {
        "frame_idx": frame_idx,
        "cam_type": cam_type,
        "sample_token": "tok",
        "scene_token": "scene",
        "boxes3d": np.zeros((0, 9), dtype=np.float32),
        "ori_labels3d": [],
        "labels3d": [],
        "calib": {
            "cam_intrinsic": np.eye(4),
            "cam2ego": np.asarray(cam2ego),
            "ego2global": np.asarray(ego2global),
        },
    }


def _override(entries):
    return {
        "schema_version": "driveloop_dd2_override.v0",
        "boxes3d": {"per_frame_append_ego": entries},
    }


def test_ego_entry_consumed_per_camera_with_distinct_boxes():
    override = _override([_ego_entry()])

    front = _sample("cam_front", CAM2EGO_FRONT, E2G_FRONT)
    updated_front, audit_front = apply_dd2_override_to_sample(front, override)
    fl = _sample("cam_front_left", CAM2EGO_FL, E2G_FL)
    updated_fl, audit_fl = apply_dd2_override_to_sample(fl, override)

    assert updated_front["boxes3d"].shape == (1, 9)
    assert updated_fl["boxes3d"].shape == (1, 9)
    # Anti-mirror sentinel through the FULL apply path: ego y=+3.5 (LEFT)
    # must land on the LEFT side of cam_front (negative camera x).
    front_box = updated_front["boxes3d"][0]
    assert front_box[2] > 15.0
    assert front_box[0] < -2.0
    # True per-view projection: the two cameras get DISTINCT boxes.
    assert not np.allclose(front_box, updated_fl["boxes3d"][0], atol=1e-3)
    assert updated_front["ori_labels3d"] == ["vehicle.motorcycle"]

    for audit in (audit_front, audit_fl):
        applied = [item for item in audit["applied"] if item.get("mode") == "per_frame_append_ego"]
        assert len(applied) == 1
        assert applied[0]["accepted_count"] == 1
        assert applied[0]["conversion"]["mode"] == "ego_entry_to_cam_box9"
        assert audit["changed"]["boxes3d"] is True


def test_ego_entry_identity_mismatch_not_applied():
    override = _override([_ego_entry(frame_idx=100)])
    other_frame = _sample("cam_front", CAM2EGO_FRONT, E2G_FRONT, frame_idx=101)
    updated, audit = apply_dd2_override_to_sample(other_frame, override)

    assert updated["boxes3d"].shape == (0, 9)
    skip = next(item for item in audit["skipped"] if item.get("mode") == "per_frame_append_ego")
    assert skip["reason"] == "no_matching_or_convertible_entries"
    assert skip["selection_skipped_entries"][0]["reason"] == "sample_identity_mismatch"


# Idealized rear camera: z_cam = -x_ego, x_cam = +y_ego, y_cam = -z_ego
# (right-handed; det=+1). A forward actor must be culled here.
CAM2EGO_BACK = [
    [0.0, 0.0, -1.0, -0.5],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 1.5],
    [0.0, 0.0, 0.0, 1.0],
]


def test_ego_entry_behind_camera_is_culled_before_dd2_depth_assert():
    override = _override(
        [
            {
                **_ego_entry(),
                "sample_identities": [
                    {"cam_type": "cam_back", "frame_idx": 100, "sample_token": "tok", "scene_token": "scene"},
                ],
            }
        ]
    )
    back = _sample("cam_back", CAM2EGO_BACK, E2G_FRONT)
    updated, audit = apply_dd2_override_to_sample(back, override)

    # The forward actor is behind cam_back: no box may be appended,
    # otherwise the DD2 transform depth assert would crash the run.
    assert updated["boxes3d"].shape == (0, 9)
    skip = next(item for item in audit["skipped"] if item.get("mode") == "per_frame_append_ego")
    culled = skip["conversion_skipped_entries"][0]
    assert culled["reason"] == "behind_camera_culled"
    assert culled["center_cam_z"] < 0.0
    assert audit["changed"]["boxes3d"] is False


def test_ego_entry_skipped_when_record_calib_missing_extrinsics():
    override = _override([_ego_entry()])
    sample = _sample("cam_front", CAM2EGO_FRONT, E2G_FRONT)
    sample["calib"] = {"cam_intrinsic": np.eye(4)}
    updated, audit = apply_dd2_override_to_sample(sample, override)

    assert updated["boxes3d"].shape == (0, 9)
    skip = next(item for item in audit["skipped"] if item.get("mode") == "per_frame_append_ego")
    assert skip["conversion_skipped_entries"][0]["reason"] == "record_calib_missing_extrinsics"


def _identities_with_calib(cam_types, frame_idx=100):
    identities = []
    for idx, cam in enumerate(cam_types):
        identity = {
            "available": True,
            "relative_step": 0,
            "frame_idx": frame_idx,
            "cam_type": cam,
            "sample_token": "tok",
            "scene_token": "scene",
            "record_index": idx,
        }
        if cam == "cam_front":
            identity["calib"] = {"cam2ego": CAM2EGO_FRONT, "ego2global": E2G_FRONT}
        elif cam == "cam_front_left":
            identity["calib"] = {"cam2ego": CAM2EGO_FL, "ego2global": E2G_FL}
        identities.append(identity)
    return identities


ALL_CAMS = [
    "cam_front_left", "cam_front", "cam_front_right",
    "cam_back_right", "cam_back", "cam_back_left",
]


def _plan_box(frame_idx=0):
    # Left cut-in draft box in the cam_front frame (x left-negative).
    return {
        "frame_idx": frame_idx,
        "actor_id": "actor_01",
        "synthetic_track_id": "actor_01_synthetic_motion_track",
        "category": "motorcycle",
        "box3d": [-3.5, 1.8, 20.0, 0.8, 1.4, 2.2, 0.0, -0.25, 0.0],
        "maneuver": "cut_in",
    }


def test_backend_maps_plan_boxes_to_one_ego_entry_per_frame(monkeypatch):
    backend = object.__new__(DriveDreamer2Backend)
    monkeypatch.setattr(
        backend,
        "_build_source_bound_sample_identities",
        lambda binding: _identities_with_calib(ALL_CAMS),
        raising=False,
    )
    mapped, mapping = backend._map_per_frame_actor_boxes_to_ego_entries([_plan_box()], {})

    assert len(mapped) == 1
    entry = mapped[0]
    assert entry["motion_surface"] == "boxes3d.per_frame_append_ego"
    assert entry["ref_ego2global"] == E2G_FRONT
    assert {i["cam_type"] for i in entry["sample_identities"]} == set(ALL_CAMS)
    # Ego lift of a LEFT actor: ego y must be positive (ego y is left).
    assert entry["ego"]["center_ego"][1] > 2.0
    assert entry["ego"]["center_ego"][0] > 15.0
    assert mapping["available"] is True
    assert mapping["mode"] == "ego_frame_one_entry_per_video_frame"
    assert mapping["mapped_entry_count"] == 1
    assert mapping["view_filter"]["filtered_out_count"] == 0


def test_backend_reports_missing_front_calib(monkeypatch):
    backend = object.__new__(DriveDreamer2Backend)
    identities = _identities_with_calib(ALL_CAMS)
    for identity in identities:
        identity.pop("calib", None)
    monkeypatch.setattr(
        backend,
        "_build_source_bound_sample_identities",
        lambda binding: identities,
        raising=False,
    )
    mapped, mapping = backend._map_per_frame_actor_boxes_to_ego_entries([_plan_box()], {})

    assert mapped == []
    assert mapping["available"] is False
    assert mapping["missing_front_calib_relative_frame_idx"] == [0]


def _minimal_plan_inputs():
    structural_input_plan = {
        "control_level": "tensor_override_runtime",
        "scene_description": {"value": "night", "source": "text_control.prompt"},
    }
    override_candidate_plan = {
        "box_synthesis_plan": {"box_synthesis_draft": {"draft_boxes3d": []}},
        "actor_motion_surface_plan": {
            "per_frame_boxes3d": [_plan_box()],
            "target_cam_types": ["cam_front", "cam_front_left"],
        },
    }
    return structural_input_plan, override_candidate_plan


def test_build_override_json_flag_on_emits_ego_surface_and_suppresses_clones(monkeypatch):
    backend = object.__new__(DriveDreamer2Backend)
    monkeypatch.setattr(
        backend,
        "_build_source_bound_sample_identities",
        lambda binding: _identities_with_calib(ALL_CAMS),
        raising=False,
    )
    monkeypatch.setenv("DRIVELOOP_EGO_INJECTION", "1")
    structural_input_plan, override_candidate_plan = _minimal_plan_inputs()
    override = backend._build_override_json("night", structural_input_plan, override_candidate_plan, {})

    boxes3d = override["boxes3d"]
    assert boxes3d["mode"] == "append_and_per_frame_append_ego"
    assert boxes3d["per_frame_append"] == []
    assert len(boxes3d["per_frame_append_ego"]) == 1
    assert boxes3d["ego_injection"]["enabled"] is True
    assert override["audit"]["control_level"] == "tensor_override_runtime"
    assert (
        "per_frame_actor_boxes3d_runtime_surface_connected_ego_frame"
        in override["audit"]["limitations"]
    )


def test_build_override_json_flag_off_keeps_legacy_surface(monkeypatch):
    backend = object.__new__(DriveDreamer2Backend)
    monkeypatch.setattr(
        backend,
        "_build_source_bound_sample_identities",
        lambda binding: _identities_with_calib(ALL_CAMS),
        raising=False,
    )
    monkeypatch.delenv("DRIVELOOP_EGO_INJECTION", raising=False)
    structural_input_plan, override_candidate_plan = _minimal_plan_inputs()
    override = backend._build_override_json("night", structural_input_plan, override_candidate_plan, {})

    boxes3d = override["boxes3d"]
    assert boxes3d["mode"] == "append_and_per_frame_append"
    assert boxes3d["per_frame_append_ego"] == []
    assert boxes3d["ego_injection"]["enabled"] is False
    assert len(boxes3d["per_frame_append"]) == 2  # view-filtered per-cam clones


def _tangent_entry(step, e2g, center):
    return {
        "relative_frame_idx": step,
        "ref_ego2global": e2g,
        "ego": {"center_ego": list(center), "dims": [0.8, 1.4, 2.2], "heading_ego": -0.25},
    }


def _translated(e2g, dx):
    out = [list(row) for row in e2g]
    out[0][3] += dx
    return out


IDENTITY_E2G = [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]


def test_tangent_heading_uses_global_motion_when_ego_advances():
    from driveloop.ego_injection import apply_trajectory_tangent_heading

    # Ego advances 5 m/frame; relative x drifts BACK 0.34 m/frame.
    # Global motion is forward (+4.66 m/frame): heading must be ~0,
    # not ~pi (which a relative-frame tangent would produce).
    entries = [
        _tangent_entry(k, _translated(IDENTITY_E2G, 5.0 * k), [20.0 - 0.34 * k, 3.5, 1.0])
        for k in range(3)
    ]
    mode = apply_trajectory_tangent_heading(entries)

    assert mode == "trajectory_tangent_global"
    for e in entries:
        assert abs(e["ego"]["heading_ego"]) < 0.1
        assert e["heading"]["mode"] == "trajectory_tangent_global"
        assert e["heading"]["plan_heading_ego"] == -0.25


def test_tangent_heading_static_ego_keeps_plan_for_tiny_displacement():
    from driveloop.ego_injection import apply_trajectory_tangent_heading

    entries = [
        _tangent_entry(k, IDENTITY_E2G, [20.0, 3.5, 1.0])
        for k in range(3)
    ]
    mode = apply_trajectory_tangent_heading(entries)

    assert mode == "plan_yaw_kept_small_displacement"
    for e in entries:
        assert e["ego"]["heading_ego"] == -0.25


def test_backend_mapping_applies_tangent_heading_and_env_disable(monkeypatch):
    def build(env_value):
        backend = object.__new__(DriveDreamer2Backend)
        identities = _identities_with_calib(ALL_CAMS, frame_idx=100) + [
            {**i, "relative_step": 1, "frame_idx": 101, "record_index": i["record_index"] + 6}
            for i in _identities_with_calib(ALL_CAMS, frame_idx=101)
        ]
        monkeypatch.setattr(
            backend, "_build_source_bound_sample_identities", lambda binding: identities, raising=False
        )
        if env_value is None:
            monkeypatch.delenv("DRIVELOOP_EGO_TANGENT_HEADING", raising=False)
        else:
            monkeypatch.setenv("DRIVELOOP_EGO_TANGENT_HEADING", env_value)
        boxes = [_plan_box(0), {**_plan_box(1), "box3d": [-3.2, 1.8, 21.0, 0.8, 1.4, 2.2, 0.0, -0.25, 0.0]}]
        return backend._map_per_frame_actor_boxes_to_ego_entries(boxes, {})

    mapped, mapping = build(None)
    assert mapping["heading_mode"] == "trajectory_tangent_global"
    assert all(e["heading"]["mode"] == "trajectory_tangent_global" for e in mapped)

    mapped_off, mapping_off = build("0")
    assert mapping_off["heading_mode"] == "plan_yaw_tangent_disabled"
    assert all("heading" not in e for e in mapped_off)


REAL_MOTO_BOX = [-9.1778, 1.1333, 18.7546, 0.498, 1.761, 0.697, 0.0, 1.5725, 0.0]


def _front_record(frame_idx, moto_box=REAL_MOTO_BOX, instance="inst-moto"):
    import numpy as np

    return {
        "cam_type": "cam_front",
        "frame_idx": frame_idx,
        "sample_token": "tok",
        "scene_token": "scene",
        "boxes3d": np.asarray(
            [[1.0, 1.5, 30.0, 1.9, 1.6, 4.5, 0.0, 0.1, 0.0], moto_box], dtype=np.float64
        ),
        "ori_labels3d": ["vehicle.car", "vehicle.motorcycle"],
        "instance_tokens": ["inst-car", instance],
        "calib": {"cam2ego": CAM2EGO_FRONT, "ego2global": E2G_FRONT},
    }


def _real_track_backend(monkeypatch, front_records, binding=None):
    backend = object.__new__(DriveDreamer2Backend)
    monkeypatch.setattr(
        backend,
        "_build_source_bound_sample_identities",
        lambda b: _identities_with_calib(ALL_CAMS, frame_idx=100),
        raising=False,
    )
    monkeypatch.setattr(
        backend,
        "_build_source_bound_front_records",
        lambda b: front_records,
        raising=False,
    )
    return backend


def test_real_track_mode_lifts_real_box_and_suppresses_synthetic(monkeypatch):
    from driveloop.ego_injection import cam_box9_to_ego_entry

    monkeypatch.delenv("DRIVELOOP_EGO_REAL_TRACK", raising=False)
    backend = _real_track_backend(
        monkeypatch,
        [{"relative_step": 0, "record_index": 30, "record": _front_record(100)}],
    )
    binding = {"ready": True, "selector": {"instance_token": "inst-moto"}}
    entries, mapping = backend._map_real_track_to_ego_entries([_plan_box(0)], binding)

    assert mapping["mode"] == "source_bound_real_track_ego"
    assert mapping["selection_basis"] == "binding_instance_token"
    assert mapping["synthetic_suppressed_count"] == 1
    assert len(entries) == 1
    entry = entries[0]
    assert entry["source"] == "source_bound_real_track"
    assert entry["actor_id"] == "inst-moto"
    assert entry["heading"]["mode"] == "real_track_annotation"
    expected = cam_box9_to_ego_entry(REAL_MOTO_BOX, CAM2EGO_FRONT)
    assert entry["ego"]["center_ego"] == expected["center_ego"]
    assert {i["cam_type"] for i in entry["sample_identities"]} == set(ALL_CAMS)


def test_real_track_mode_env_disable_and_fallback(monkeypatch):
    backend = _real_track_backend(
        monkeypatch,
        [{"relative_step": 0, "record_index": 30, "record": _front_record(100)}],
    )
    monkeypatch.setenv("DRIVELOOP_EGO_REAL_TRACK", "0")
    entries, mapping = backend._map_real_track_to_ego_entries([_plan_box(0)], {"ready": True})
    assert entries == []
    assert mapping["reason"] == "real_track_mode_disabled"

    monkeypatch.delenv("DRIVELOOP_EGO_REAL_TRACK", raising=False)
    record = _front_record(100)
    record["ori_labels3d"] = ["vehicle.car", "vehicle.car"]
    backend2 = _real_track_backend(
        monkeypatch,
        [{"relative_step": 0, "record_index": 30, "record": record}],
    )
    entries2, mapping2 = backend2._map_real_track_to_ego_entries([_plan_box(0)], {"ready": True})
    assert entries2 == []
    assert mapping2["reason"] == "no_real_track_boxes_for_category"


def test_real_track_entry_roundtrips_to_original_annotation(monkeypatch):
    import numpy as np

    monkeypatch.delenv("DRIVELOOP_EGO_REAL_TRACK", raising=False)
    backend = _real_track_backend(
        monkeypatch,
        [{"relative_step": 0, "record_index": 30, "record": _front_record(100)}],
    )
    entries, _ = backend._map_real_track_to_ego_entries([_plan_box(0)], {"ready": True})
    override = _override(entries)
    front = _sample("cam_front", CAM2EGO_FRONT, E2G_FRONT)
    updated, audit = apply_dd2_override_to_sample(front, override)

    assert updated["boxes3d"].shape == (1, 9)
    assert np.allclose(updated["boxes3d"][0][:6], REAL_MOTO_BOX[:6], atol=1e-5)
    applied = [a for a in audit["applied"] if a.get("mode") == "per_frame_append_ego"]
    assert len(applied) == 1


def test_emission_to_consumption_roundtrip_reproduces_cam_front_plan_box(monkeypatch):
    backend = object.__new__(DriveDreamer2Backend)
    monkeypatch.setattr(
        backend,
        "_build_source_bound_sample_identities",
        lambda binding: _identities_with_calib(ALL_CAMS),
        raising=False,
    )
    plan_box = _plan_box()
    mapped, _ = backend._map_per_frame_actor_boxes_to_ego_entries([plan_box], {})
    override = _override(mapped)

    front = _sample("cam_front", CAM2EGO_FRONT, E2G_FRONT)
    updated, _ = apply_dd2_override_to_sample(front, override)

    assert updated["boxes3d"].shape == (1, 9)
    # cam_front consumption uses the same record whose calib lifted the
    # plan box: the round trip must reproduce the original draft box.
    assert np.allclose(updated["boxes3d"][0][:6], plan_box["box3d"][:6], atol=1e-5)
    assert abs(float(updated["boxes3d"][0][7]) - plan_box["box3d"][7]) < 1e-3


def test_ego_entry_outside_camera_fov_is_culled(monkeypatch):
    # Actor 20 m ahead, 3.5 m LEFT: bearing ~10 deg, ~45 deg off the
    # front-left camera axis - its CENTER projects outside that image,
    # leaving only clipped corner fragments on the canvas (junk edge
    # conditioning; human-reviewed teleport artifact, 2026-07-09).
    monkeypatch.delenv("DRIVELOOP_EGO_FOV_CULL", raising=False)
    override = _override([_ego_entry()])
    fl = _sample("cam_front_left", CAM2EGO_FL, E2G_FL)
    fl["calib"]["cam_intrinsic"] = [
        [1266.4, 0.0, 800.0, 0.0],
        [0.0, 1266.4, 450.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    updated, audit = apply_dd2_override_to_sample(fl, override)

    assert updated["boxes3d"].shape == (0, 9)
    skip = next(item for item in audit["skipped"] if item.get("mode") == "per_frame_append_ego")
    culled = skip["conversion_skipped_entries"][0]
    assert culled["reason"] == "center_outside_image_culled"

    # A/B escape hatch: disabling the cull restores the old behavior.
    monkeypatch.setenv("DRIVELOOP_EGO_FOV_CULL", "0")
    fl2 = _sample("cam_front_left", CAM2EGO_FL, E2G_FL)
    updated2, _ = apply_dd2_override_to_sample(fl2, _override([_ego_entry()]))
    assert updated2["boxes3d"].shape == (1, 9)


def test_ego_entry_centered_in_front_camera_survives_fov_cull(monkeypatch):
    monkeypatch.delenv("DRIVELOOP_EGO_FOV_CULL", raising=False)
    front = _sample("cam_front", CAM2EGO_FRONT, E2G_FRONT)
    front["calib"]["cam_intrinsic"] = [
        [1266.4, 0.0, 800.0, 0.0],
        [0.0, 1266.4, 450.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    updated, audit = apply_dd2_override_to_sample(front, _override([_ego_entry()]))

    assert updated["boxes3d"].shape == (1, 9)
    applied = [a for a in audit["applied"] if a.get("mode") == "per_frame_append_ego"]
    assert applied and applied[0]["accepted_count"] == 1


def test_real_track_never_labels_another_actor_with_requested_identity(monkeypatch):
    backend = _real_track_backend(monkeypatch, [
        {'relative_step': 0, 'record_index': 30, 'record': _front_record(100, instance='other-moto')}
    ])
    entries, mapping = backend._map_real_track_to_ego_entries(
        [_plan_box(0)], {'ready': True, 'selector': {'instance_token': 'inst-moto'}})
    assert entries == []
    assert mapping['missing_relative_steps'] == [0]


def test_real_track_cannot_verify_requested_actor_without_tokens(monkeypatch):
    record = _front_record(100)
    del record['instance_tokens']
    backend = _real_track_backend(monkeypatch, [{'relative_step': 0, 'record_index': 30, 'record': record}])
    entries, mapping = backend._map_real_track_to_ego_entries(
        [_plan_box(0)], {'ready': True, 'selector': {'instance_token': 'inst-moto'}})
    assert entries == []
    assert mapping['available'] is False
