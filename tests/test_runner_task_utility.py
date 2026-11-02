from driveloop.backends.mock import MockGenerationBackend
from driveloop.evaluators import BaseEvaluator
from driveloop.runner import DriveLoopRunner
from driveloop.schema import Diagnosis, DriveLoopConfig, DriveLoopRequest, Evaluation


class FixedScoreEvaluator(BaseEvaluator):
    def __init__(self, score: float) -> None:
        self.score = score

    def evaluate(self, generation) -> Evaluation:
        return Evaluation(self.score, {}, Diagnosis(False, ["low_detection_coverage"], []))


REQUEST = DriveLoopRequest(prompt="at night a motorcycle cuts in from the left lane")


def _runner(tmp_path, **config_kwargs):
    config = DriveLoopConfig(max_iterations=2, output_dir=tmp_path, **config_kwargs)
    return DriveLoopRunner(
        backend=MockGenerationBackend(output_dir=tmp_path / "mock"),
        evaluator=FixedScoreEvaluator(0.468199),
        config=config,
    )


def test_utility_disabled_by_default(tmp_path):
    result = _runner(tmp_path).run(REQUEST)
    assert result.best_evaluation.score == 0.468199
    assert "J" not in result.best_evaluation.metrics


def test_utility_enabled_replaces_acceptance_score(tmp_path):
    result = _runner(tmp_path, use_task_utility=True).run(REQUEST)
    metrics = result.best_evaluation.metrics
    for key in ("J", "S_perc", "S_ctrl", "S_intent"):
        assert key in metrics
    assert metrics["S_perc"] == 0.468199
    assert result.best_evaluation.score == metrics["J"]


def test_utility_weights_override(tmp_path):
    result = _runner(
        tmp_path,
        use_task_utility=True,
        utility_weights={"perception": 1.0, "control": 0.0, "intent": 0.0},
    ).run(REQUEST)
    metrics = result.best_evaluation.metrics
    assert abs(metrics["J"] - metrics["S_perc"]) < 1e-9


def test_high_utility_does_not_override_failed_diagnosis(tmp_path):
    config = DriveLoopConfig(
        max_iterations=3, output_dir=tmp_path, target_score=0.1, use_task_utility=True
    )
    runner = DriveLoopRunner(
        backend=MockGenerationBackend(output_dir=tmp_path / "mock"),
        evaluator=FixedScoreEvaluator(0.9),
        config=config,
    )
    result = runner.run(REQUEST)
    assert len(result.attempt_history) == 3
    assert result.return_status == "budget_exhausted"
    assert all(a.status != "accepted" for a in result.attempt_history)


class _UnmeasurableEvaluator(BaseEvaluator):
    # Mimics v10b when the maneuver view restriction resolves to no scorable
    # view: score 0, not passed, but flagged view_restriction_unresolved.
    def evaluate(self, generation) -> Evaluation:
        return Evaluation(
            0.0,
            {"perception_measured": 1.0, "perception_view_restriction_unresolved": 1.0},
            Diagnosis(False, ["no_maneuver_view_restriction_resolvable"], []),
        )


def test_unmeasurable_case_stops_the_loop_and_is_flagged(tmp_path):
    # An unmeasurable case must not churn refinements against something that
    # cannot be measured, and must be recorded as unmeasurable rather than a
    # low-score failure.
    config = DriveLoopConfig(max_iterations=3, output_dir=tmp_path, target_score=0.8)
    runner = DriveLoopRunner(
        backend=MockGenerationBackend(output_dir=tmp_path / "mock"),
        evaluator=_UnmeasurableEvaluator(),
        config=config,
    )
    result = runner.run(REQUEST)
    assert len(result.attempt_history) == 1
    assert result.attempt_history[-1].status == "perception_unmeasurable"


def test_low_score_still_churns_when_measurable(tmp_path):
    # Backward compatibility: a plain low score with no unresolved flag (v9)
    # still drives refinement to max_iterations.
    config = DriveLoopConfig(max_iterations=3, output_dir=tmp_path, target_score=0.8)
    runner = DriveLoopRunner(
        backend=MockGenerationBackend(output_dir=tmp_path / "mock"),
        evaluator=FixedScoreEvaluator(0.1),
        config=config,
    )
    result = runner.run(REQUEST)
    assert len(result.attempt_history) == 3
    assert result.attempt_history[-1].status == "needs_refinement"
