"""Acceptance is separate from the scalar used to rank candidates.

The joint policy implements the paper's S_perc/direction gate. A missing
direction measurement is not a pass. The legacy OR policy is available only
to reproduce historical runs; it must not be called a joint gate.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

POLICIES = ("score_and_diagnosis", "joint_perception_direction", "legacy_score_or_diagnosis")


def finite_number(value: Any):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def joint_gate(metrics: Mapping[str, Any], threshold: float, raw_score=None) -> str:
    """raw_score may be supplied only before task-utility replacement."""
    score = finite_number(metrics.get("S_perc", raw_score))
    if score is None:
        return "UNMEASURED"
    if score < threshold:
        return "LOW_SPERC"
    direction = finite_number(metrics.get("maneuver_direction_consistent"))
    if direction is None:
        return "UNMEASURED"
    return "PASS" if direction == 1.0 else "DIR_FAIL"


def acceptance_state(evaluation, config) -> str:
    policy = config.acceptance_policy
    if policy not in POLICIES:
        raise ValueError("unknown acceptance_policy: %r" % policy)
    if policy == "joint_perception_direction":
        raw = None if config.use_task_utility else evaluation.score
        return joint_gate(evaluation.metrics, config.perception_acceptance_threshold, raw)
    score = finite_number(evaluation.score)
    if score is None:
        return "UNMEASURED"
    enough = score >= config.target_score
    if policy == "legacy_score_or_diagnosis":
        return "PASS" if enough or evaluation.diagnosis.passed else "FAIL"
    direction = evaluation.metrics.get("maneuver_direction_consistent")
    if direction is not None and finite_number(direction) != 1.0:
        return "DIR_FAIL"
    return "PASS" if enough and evaluation.diagnosis.passed else "FAIL"
