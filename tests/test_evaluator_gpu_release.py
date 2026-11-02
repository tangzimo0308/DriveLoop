"""The experiment driver must not hold GPU memory across DD2
generations: a detector-held 1.5 GiB caused CUDA OOM in the UNet after
several closed-loop iterations on a 22 GiB A10 (2026-07-09)."""
import tempfile
from pathlib import Path

from driveloop import DriveLoopConfig, DriveLoopRequest, DriveLoopRunner
from driveloop.backends.base import GenerationBackend
from driveloop.evaluator import BaseEvaluator
from driveloop.perception_video import UltralyticsYOLODetector
from driveloop.schema import Evaluation, Generation


def test_yolo_detector_release_gpu_drops_model_and_detect_reloads():
    detector = object.__new__(UltralyticsYOLODetector)
    detector.confidence_threshold = 0.25
    detector.inference_confidence_threshold = 0.25
    detector.begin_runtime_audit()
    detector.weights = "fake.pt"
    reloads = []

    class _FakeModel:
        def predict(self, source, verbose, conf):
            assert conf == 0.25
            return []

    detector._load_model = lambda: reloads.append(1) or _FakeModel()
    detector.model = _FakeModel()

    detector.release_gpu()  # must not raise
    assert detector.model is None

    assert detector.detect("frame.jpg", 0) == []
    assert reloads == [1]  # lazily reloaded exactly once


class _ReleaseTrackingDetector:
    def __init__(self):
        self.release_calls = 0

    def release_gpu(self):
        self.release_calls += 1


class _DetectorBackedEvaluator(BaseEvaluator):
    def __init__(self, detector):
        self.detector = detector

    def evaluate(self, generation: Generation) -> Evaluation:
        return Evaluation(score=0.0)


class _NullBackend(GenerationBackend):
    def generate(self, request, iteration):
        return Generation(iteration=iteration, prompt=request.prompt, artifacts={}, metadata={})


def test_runner_releases_detector_gpu_after_each_evaluation():
    detector = _ReleaseTrackingDetector()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = DriveLoopRunner(
            backend=_NullBackend(),
            evaluator=_DetectorBackedEvaluator(detector),
            config=DriveLoopConfig(
                max_iterations=2,
                target_score=0.9,
                output_dir=Path(tmpdir),
            ),
        ).run(DriveLoopRequest(prompt="release test"))

    assert detector.release_calls == len(result.attempt_history)
    assert detector.release_calls >= 1
