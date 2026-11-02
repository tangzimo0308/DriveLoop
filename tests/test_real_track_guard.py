import copy

import numpy as np
import pytest

from driveloop.real_track_guard import check_real_track


def trajectory(ys, ego_ys=None):
    entries = []
    for i, y in enumerate(ys):
        transform = np.eye(4)
        transform[1, 3] = (ego_ys or [0] * len(ys))[i]
        entries.append({"relative_frame_idx": i, "actor_id": "same-instance",
                        "ego": {"center_ego": [20, y, 0]},
                        "ref_ego2global": transform.tolist()})
    return entries


@pytest.mark.parametrize("side", [-1, 1])
def test_matching_left_and_right_tracks(side):
    plan = trajectory([side * y for y in [3, 2, 1]])
    assert check_real_track(plan, plan, "cut_in")["compatible"]
    wrong = trajectory([-side * y for y in [3, 2, 1]])
    result = check_real_track(wrong, plan, "cut_in")
    assert result["reason"] == "source_track_wrong_initial_side"


@pytest.mark.parametrize("ys", [[3, 3, 3], [3, 4, 5]])
def test_stationary_or_outward_track_rejected(ys):
    result = check_real_track(trajectory(ys), trajectory([3, 2, 1]), "cut_in")
    assert result["reason"] == "source_track_no_relative_inward_motion"


def test_ego_motion_does_not_turn_parked_actor_into_cut_in():
    # Actor stays at world y=3 while ego translates left.
    real = trajectory([3, 2, 1], ego_ys=[0, 1, 2])
    result = check_real_track(real, trajectory([3, 2, 1]), "cut_in")
    assert result["reason"] == "source_track_no_independent_inward_motion"
    assert result["fixed_frame_inward_meters"] == 0


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "nan", "singular", "identity"])
def test_unverifiable_track_is_rejected(mutation):
    plan = trajectory([3, 2, 1])
    real = copy.deepcopy(plan)
    if mutation == "missing":
        real.pop()
    elif mutation == "duplicate":
        real[-1]["relative_frame_idx"] = 1
    elif mutation == "nan":
        real[0]["ego"]["center_ego"][1] = float("nan")
    elif mutation == "singular":
        real[0]["ref_ego2global"] = np.zeros((4, 4)).tolist()
    else:
        real[-1]["actor_id"] = "other"
    assert not check_real_track(real, plan, "cut_in")["compatible"]


def test_backend_rejection_keeps_plan_and_audits_reason(monkeypatch):
    from driveloop.backends.drivedreamer2 import DriveDreamer2Backend
    backend = object.__new__(DriveDreamer2Backend)
    plan = trajectory([3, 2, 1])
    monkeypatch.setenv("DRIVELOOP_EGO_INJECTION", "1")
    monkeypatch.setattr(backend, "_map_real_track_to_ego_entries", lambda *a: (
        trajectory([-3, -2, -1]), {"available": True, "selection_basis": "binding_instance_token"}))
    monkeypatch.setattr(backend, "_map_per_frame_actor_boxes_to_ego_entries", lambda *a: (
        plan, {"available": True, "mode": "ego_frame_one_entry_per_video_frame"}))
    candidate = {"actor_motion_surface_plan": {"maneuver": "cut_in",
                 "per_frame_boxes3d": [{"frame_idx": i, "category": "bus", "maneuver": "cut_in"} for i in range(3)]}}
    output = backend._build_override_json("bus from left", {"control_level": "tensor_override_runtime"}, candidate, {})
    assert output["boxes3d"]["per_frame_append_ego"] == plan
    mapping = output["audit"]["actor_motion_frame_mapping"]
    assert mapping["real_track_fallback_reason"] == "source_track_wrong_initial_side"
    assert not mapping["rejected_real_track_audit"]["trajectory_compatibility"]["compatible"]
    monkeypatch.setenv("DRIVELOOP_EGO_REQUIRE_REAL_TRACK", "1")
    with pytest.raises(RuntimeError, match="source_track_wrong_initial_side"):
        backend._build_override_json("bus from left", {"control_level": "tensor_override_runtime"}, candidate, {})
