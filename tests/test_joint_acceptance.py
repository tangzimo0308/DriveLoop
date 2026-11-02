import pytest

from driveloop.acceptance import joint_gate
from driveloop.runner import DriveLoopRunner
from driveloop.schema import DriveLoopConfig, DriveLoopRequest, Generation, Evaluation, Diagnosis, Refinement


def evaluation(score, direction, passed=False, sperc=None):
    metrics = {"S_perc": score if sperc is None else sperc}
    if direction is not None:
        metrics["maneuver_direction_consistent"] = direction
    return Evaluation(score, metrics, Diagnosis(passed, [] if passed else ["maneuver_direction_mismatch"]))


class Backend:
    def generate(self, request, iteration):
        return Generation(iteration, request.prompt)


class Evaluator:
    def __init__(self, evaluations):
        self.evaluations = evaluations

    def evaluate(self, generation):
        return self.evaluations[generation.iteration]


class Refiner:
    def refine(self, request, evaluation):
        return Refinement(request.prompt, request.condition)


def run(tmp_path, evaluations, **kwargs):
    options = dict(max_iterations=len(evaluations), output_dir=tmp_path,
                   acceptance_policy="joint_perception_direction")
    options.update(kwargs)
    return DriveLoopRunner(Backend(), evaluator=Evaluator(evaluations),
                           refiner=Refiner(), config=DriveLoopConfig(**options)).run(
        DriveLoopRequest(prompt="a truck cuts in from my left"))


def test_wrong_direction_cannot_short_circuit_and_returned_is_not_rank_best(tmp_path):
    result = run(tmp_path, [evaluation(.9, 0), evaluation(.6, 1), evaluation(.95, 0)])
    assert len(result.history) == 2
    assert result.best_generation.iteration == 0
    assert result.returned_generation.iteration == 1
    assert result.return_status == "accepted"
    assert [a.status for a in result.attempt_history] == ["needs_refinement", "accepted"]


def test_fixed_budget_returns_best_passing_candidate_not_highest_wrong_candidate(tmp_path):
    result = run(tmp_path, [evaluation(.9, 0), evaluation(.6, 1), evaluation(.95, 0)], stop_on_acceptance=False)
    assert len(result.history) == 3
    assert result.best_generation.iteration == 2
    assert result.returned_generation.iteration == 1
    assert result.return_status == "accepted"


def test_high_utility_is_not_high_perception(tmp_path):
    result = run(tmp_path, [evaluation(.9, 1, sperc=.1)])
    assert result.return_status == "budget_exhausted"


def test_missing_direction_requires_review_not_acceptance(tmp_path):
    result = run(tmp_path, [evaluation(.9, None, passed=True)])
    assert result.return_status == "requires_review"
    assert result.attempt_history[0].status == "requires_review"


def test_low_perception_is_not_unmeasured(tmp_path):
    result = run(tmp_path, [evaluation(.0, None)])
    assert result.return_status == "budget_exhausted"


def test_default_gate_does_not_override_failed_diagnosis(tmp_path):
    result = run(tmp_path, [evaluation(.9, 0)], acceptance_policy="score_and_diagnosis")
    assert result.return_status != "accepted"


def test_legacy_is_explicit_and_recorded(tmp_path):
    result = run(tmp_path, [evaluation(.9, 0)], acceptance_policy="legacy_score_or_diagnosis")
    assert result.return_status == "accepted"
    assert result.acceptance_policy == "legacy_score_or_diagnosis"


def test_unavailable_binding_cannot_be_accepted(tmp_path):
    runner = DriveLoopRunner(Backend(), config=DriveLoopConfig(output_dir=tmp_path, acceptance_policy="joint_perception_direction"))
    generation = Generation(0, "truck", metadata={"dd2_source_sample_binding": {"requested": True, "ready": False}})
    assert runner._attempt_status(generation, evaluation(.9, 1), {}) == "source_binding_unavailable"


@pytest.mark.parametrize("metrics,expected", [
    ({"S_perc": .3, "maneuver_direction_consistent": 1}, "PASS"),
    ({"S_perc": .299, "maneuver_direction_consistent": 1}, "LOW_SPERC"),
    ({"S_perc": .9, "maneuver_direction_consistent": 0}, "DIR_FAIL"),
    ({"S_perc": .9}, "UNMEASURED"),
    ({"S_perc": float("nan"), "maneuver_direction_consistent": 1}, "UNMEASURED"),
    ({"J": .99, "maneuver_direction_consistent": 1}, "UNMEASURED"),
])
def test_gate_boundaries(metrics, expected):
    assert joint_gate(metrics, .3) == expected


def test_wrapper_passes_explicit_protocol():
    from argparse import Namespace
    from scripts.render_window_case import build_experiment_argv
    args = Namespace(cases="m.json", output_dir="new_run", config_name="mini", max_iterations=4,
                     target_score=.99, perception_weights=None, use_task_utility=True,
                     acceptance_policy="joint_perception_direction", perception_acceptance_threshold=.3,
                     fixed_budget=True)
    command = build_experiment_argv({"source_candidate_id": "c1", "dataset_dir": "data"}, None, args)
    assert command[command.index("--acceptance-policy") + 1] == "joint_perception_direction"
    assert "--fixed-budget" in command


def test_pipeline_summary_describes_actual_return(tmp_path, monkeypatch):
    from driveloop.experiment_pipeline import ExperimentPipeline, ExperimentPipelineConfig, ExperimentCase
    evaluations = [evaluation(.9, 0), evaluation(.6, 1)]
    result = run(tmp_path / "probe", evaluations)
    monkeypatch.setattr(DriveLoopRunner, "run", lambda self, request: result)
    pipeline = ExperimentPipeline(tmp_path / "experiment", config=ExperimentPipelineConfig())
    summary = pipeline.run_cases([ExperimentCase(name="case", prompt="truck cut in")])
    case = summary["cases"][0]
    assert case["status"] == "accepted"
    assert case["best_score"] == .9
    assert case["returned_score"] == .6
    assert case["returned_iteration"] == 1
    assert case["returned_metrics"]["maneuver_direction_consistent"] == 1
    assert case["return_status"] == "accepted"
