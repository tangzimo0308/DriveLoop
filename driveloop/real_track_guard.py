"""Conservative geometric filter for replacing a requested lateral trajectory.

This checks necessary geometric evidence, not lane topology or video semantics.
Thresholds are fixed engineering tolerances, not tuned on generated scores.
"""
from __future__ import annotations

import numpy as np

GUARD_VERSION = "lateral_geometry_v1"
MIN_LATERAL_METERS = 0.5


def check_real_track(real_entries, planned_entries, maneuver):
    audit = {
        "version": GUARD_VERSION, "maneuver": maneuver,
        "minimum_lateral_meters": MIN_LATERAL_METERS,
        "claim_boundary": "Necessary geometry only; not proof of lane entry or video success.",
    }

    def result(ok, reason, **details):
        return {**audit, "compatible": ok, "reason": reason, **details}

    if maneuver not in {"cut_in", "lane_change"}:
        return result(True, "not_a_lateral_maneuver")
    try:
        real = sorted(real_entries, key=lambda e: e["relative_frame_idx"])
        plan = sorted(planned_entries, key=lambda e: e["relative_frame_idx"])
        real_steps = [e["relative_frame_idx"] for e in real]
        plan_steps = [e["relative_frame_idx"] for e in plan]
        if len(plan) < 3 or real_steps != plan_steps or len(set(real_steps)) != len(real_steps):
            return result(False, "incomplete_or_ambiguous_trajectory")
        if len({e.get("actor_id") for e in real}) != 1:
            return result(False, "inconsistent_actor_identity")
        centers = np.asarray([e["ego"]["center_ego"] for e in real], dtype=float)
        planned = np.asarray([e["ego"]["center_ego"] for e in plan], dtype=float)
        transforms = np.asarray([e["ref_ego2global"] for e in real], dtype=float)
        if centers.shape != (len(real), 3) or planned.shape != centers.shape or transforms.shape != (len(real), 4, 4):
            return result(False, "invalid_geometry")
        if not all(np.isfinite(a).all() for a in (centers, planned, transforms)):
            return result(False, "invalid_geometry")
        if not np.allclose(transforms[:, 3, :], [0, 0, 0, 1]):
            return result(False, "invalid_geometry")
        # Ego coordinates: x forward, y left. Do not use camera-x signs here.
        side = float(np.sign(planned[0, 1]))
        details = {"requested_ego_side": side, "real_initial_ego_y": float(centers[0, 1])}
        if abs(planned[0, 1]) < MIN_LATERAL_METERS:
            return result(False, "requested_side_unverifiable", **details)
        if side * centers[0, 1] < MIN_LATERAL_METERS:
            return result(False, "source_track_wrong_initial_side", **details)
        if np.any(centers[:, 0] <= 0):
            return result(False, "source_track_not_ahead", **details)
        relative_progress = side * (centers[0, 1] - centers[-1, 1])
        # Express global positions in the first ego frame: ego translation/rotation
        # must not make a stationary source actor appear to perform a cut-in.
        world = np.einsum("nij,nj->ni", transforms, np.c_[centers, np.ones(len(real))])
        fixed = (np.linalg.inv(transforms[0]) @ world.T).T
        world_progress = side * (fixed[0, 1] - fixed[-1, 1])
        details.update(relative_inward_meters=float(relative_progress),
                       fixed_frame_inward_meters=float(world_progress))
        if relative_progress < MIN_LATERAL_METERS:
            return result(False, "source_track_no_relative_inward_motion", **details)
        if world_progress < MIN_LATERAL_METERS:
            return result(False, "source_track_no_independent_inward_motion", **details)
        return result(True, "necessary_lateral_geometry_satisfied", **details)
    except (KeyError, TypeError, ValueError, IndexError, np.linalg.LinAlgError):
        return result(False, "invalid_geometry")
