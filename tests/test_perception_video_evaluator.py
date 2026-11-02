from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from driveloop.perception_video import Detection, PerceptionVideoEvaluator, SimpleIoUTracker, UltralyticsYOLODetector
from driveloop.schema import Generation
from scripts.run_perception_video_eval import build_evaluator, build_generation


def detection(frame_index: int, confidence: float = 0.9, box=None, label: str = "motorcycle"):
    return {
        "frame_index": frame_index,
        "label": label,
        "confidence": confidence,
        "box": box or [10 + frame_index, 10, 30 + frame_index, 30],
    }


def test_iou_tracker_preserves_dominant_identity():
    tracker = SimpleIoUTracker(iou_threshold=0.3)
    tracker.update([Detection.from_dict(detection(0))])
    tracker.update([Detection.from_dict(detection(1))])
    tracker.update([Detection.from_dict(detection(2))])

    assert len(tracker.tracks) == 1
    assert len(tracker.tracks[0].detections) == 3


def test_perception_video_evaluator_computes_eq15_metrics_from_metadata():
    generation = Generation(
        iteration=0,
        prompt="night urban road with a visible motorcycle cut-in",
        artifacts={"video": "candidate.mp4"},
        metadata={
            "target_labels": ["motorcycle"],
            "perception_detections": {
                "frame_count": 4,
                "frames": [
                    {"frame_index": 0, "detections": [detection(0)]},
                    {"frame_index": 1, "detections": [detection(1)]},
                    {"frame_index": 2, "detections": [detection(2)]},
                    {"frame_index": 3, "detections": [detection(3)]},
                ],
            },
        },
    )

    evaluation = PerceptionVideoEvaluator(target_labels=["motorcycle"], pass_threshold=0.8).evaluate(generation)

    assert evaluation.diagnosis.passed is True
    assert evaluation.metrics["perception_measured"] == 1.0
    assert evaluation.metrics["Q_cov"] == 1.0
    assert evaluation.metrics["Q_track"] == 1.0
    assert evaluation.metrics["Q_id"] == 1.0
    assert evaluation.metrics["Q_box"] >= 0.8


def test_perception_video_evaluator_reports_failed_visibility():
    generation = Generation(
        iteration=0,
        prompt="night urban road with a visible motorcycle cut-in",
        metadata={
            "target_labels": ["motorcycle"],
            "perception_detections": {
                "frame_count": 4,
                "frames": [{"frame_index": 0, "detections": [detection(0, confidence=0.4)]}],
            },
        },
    )

    evaluation = PerceptionVideoEvaluator(target_labels=["motorcycle"], pass_threshold=0.8).evaluate(generation)

    assert evaluation.diagnosis.passed is False
    assert evaluation.metrics["Q_cov"] == 0.25
    assert "low_detection_coverage" in evaluation.diagnosis.reasons
    assert "unstable_track_coverage" in evaluation.diagnosis.reasons


def test_perception_video_evaluator_is_not_measured_without_detector_or_report():
    generation = Generation(
        iteration=0,
        prompt="night urban road with a visible motorcycle cut-in",
        artifacts={"video": "candidate.mp4"},
    )

    evaluation = PerceptionVideoEvaluator(target_labels=["motorcycle"]).evaluate(generation)

    assert evaluation.score == 0.0
    assert evaluation.diagnosis.passed is False
    assert "perception_detector_not_configured" in evaluation.diagnosis.reasons


def test_perception_video_eval_script_loads_detection_json(tmp_path: Path):
    detections_path = tmp_path / "detections.json"
    detections_path.write_text(
        json.dumps(
            {
                "frame_count": 2,
                "frames": [
                    {"frame_index": 0, "detections": [detection(0)]},
                    {"frame_index": 1, "detections": [detection(1)]},
                ],
            }
        ),
        encoding="utf-8",
    )

    args = Namespace(
        prompt="visible motorcycle",
        scenario_id="unit",
        video_path=None,
        detections_json=str(detections_path),
        target_label=["motorcycle"],
        yolo_weights=None,
        output_dir=str(tmp_path),
        pass_threshold=0.8,
        confidence_threshold=0.25,
        max_frames=None,
    )

    generation = build_generation(args)
    report = build_evaluator(args).build_report(generation)

    assert report["schema_version"] == "driveloop_perception_video_eval.v0"
    assert report["interpretation"]["perception_claim"] == "measured_passed"
    assert report["interpretation"]["semantic_success_claim"] == "not_proven_by_perception_metrics_alone"

def test_ultralytics_detector_materializes_frame_to_temp_image_path(monkeypatch):
    predicted_sources = []

    def fake_imwrite(path, frame):
        Path(path).write_bytes(b"fake image")
        return True

    fake_cv2 = ModuleType("cv2")
    fake_cv2.imwrite = fake_imwrite

    class FakeYOLO:
        def __init__(self, weights):
            self.weights = weights

        def predict(self, source, verbose=False, conf=None):
            assert conf == 0.25
            predicted_sources.append(source)
            assert Path(source).exists()
            return [SimpleNamespace(names={}, boxes=[])]

    fake_ultralytics = ModuleType("ultralytics")
    fake_ultralytics.YOLO = FakeYOLO

    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)

    detector = UltralyticsYOLODetector("fake-yolov8n.pt")
    detections = detector.detect(object(), frame_index=0)

    assert detections == []
    assert predicted_sources
    assert predicted_sources[0].endswith(".jpg")
    assert not Path(predicted_sources[0]).exists()
