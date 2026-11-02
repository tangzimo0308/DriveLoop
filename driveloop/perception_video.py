from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import tempfile
import copy
import hashlib
import importlib.metadata
import inspect
import json
import math
from typing import Any, Dict, Iterable, List, Protocol, Sequence, Tuple

from driveloop.evaluators import BaseEvaluator
from driveloop.schema import Diagnosis, Evaluation, Generation


Box = Tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    frame_index: int
    label: str
    confidence: float
    box: Box
    track_id: int | None = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any], frame_index: int | None = None) -> "Detection":
        box = payload.get("box") or payload.get("bbox") or payload.get("xyxy")
        if not isinstance(box, Sequence) or len(box) != 4:
            raise ValueError("detection box must have four xyxy values")
        return cls(
            frame_index=int(payload.get("frame_index", 0 if frame_index is None else frame_index)),
            label=str(payload.get("label") or payload.get("class_name") or payload.get("category") or "unknown"),
            confidence=float(payload.get("confidence", payload.get("score", 0.0))),
            box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            track_id=int(payload["track_id"]) if payload.get("track_id") is not None else None,
        )

    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def iou(self, other: "Detection") -> float:
        ax1, ay1, ax2, ay2 = self.box
        bx1, by1, bx2, by2 = other.box
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area() + other.area() - inter
        return 0.0 if union <= 0 else inter / union


@dataclass
class Track:
    track_id: int
    label: str
    detections: List[Detection] = field(default_factory=list)

    @property
    def last_frame(self) -> int:
        return self.detections[-1].frame_index

    def append(self, detection: Detection) -> None:
        self.detections.append(detection)


class VideoDetector(Protocol):
    def detect(self, frame: Any, frame_index: int) -> List[Detection]:
        ...


class VideoFrameReader(Protocol):
    def read(self, video_path: Path, max_frames: int | None = None) -> List[Any]:
        ...


class OpenCVFrameReader:
    def read(self, video_path: Path, max_frames: int | None = None) -> List[Any]:
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for pixel video evaluation") from exc

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"could not open video artifact: {video_path}")

        frames: List[Any] = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
                if max_frames is not None and len(frames) >= max_frames:
                    break
        finally:
            capture.release()
        return frames


class UltralyticsYOLODetector:
    """YOLO prediction with separately specified inference and post-filter thresholds.

    confidence_threshold retains its legacy meaning: filtering returned boxes.
    The upstream default is explicitly 0.25 to preserve the audited protocol.
    Changing inference_confidence_threshold defines a different measurement
    configuration; it must not silently replace archived experiment results.
    """

    def __init__(
        self, weights: str | Path, confidence_threshold: float = 0.25,
        *, inference_confidence_threshold: float = 0.25,
    ) -> None:
        for name, value in (("confidence_threshold", confidence_threshold),
                            ("inference_confidence_threshold", inference_confidence_threshold)):
            if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        self.confidence_threshold = float(confidence_threshold)
        self.inference_confidence_threshold = float(inference_confidence_threshold)
        self.weights = str(weights)
        self.begin_runtime_audit()
        self.model = self._load_model()
        self._snapshot_model_identity()

    @staticmethod
    def _file_sha256(path) -> str | None:
        path = Path(path)
        if not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _snapshot_model_identity(self) -> None:
        # Called after each load, rather than hashing the weights for every frame.
        weight_path = Path(getattr(self.model, "ckpt_path", None) or self.weights).resolve()
        try:
            version = importlib.metadata.version("ultralytics")
        except importlib.metadata.PackageNotFoundError:
            version = None
        try:
            predict_path = inspect.getsourcefile(self.model.predict)
        except (TypeError, OSError):
            predict_path = None
        self._model_identity = {
            "ultralytics_version": version,
            "weights_path": str(weight_path),
            "weights_sha256": self._file_sha256(weight_path),
            "predict_source_path": predict_path,
            "predict_source_sha256": self._file_sha256(predict_path) if predict_path else None,
            "adapter_source_sha256": self._file_sha256(__file__),
        }

    def begin_runtime_audit(self) -> None:
        """Start a fresh evaluation scope; do not reuse the prior attempt's calls."""
        self._runtime_variants = []
        self._successful_predict_calls = 0

    def _record_runtime(self) -> None:
        predictor = getattr(self.model, "predictor", None)
        args = getattr(predictor, "args", None)
        keys = ("conf", "iou", "imgsz", "device", "half", "augment", "max_det",
                "rect", "agnostic_nms", "classes", "task", "mode", "batch",
                "save", "dnn", "compile", "nms", "channels_last", "quantize")
        effective = None if args is None else {
            key: args.get(key) if isinstance(args, dict) else getattr(args, key, None)
            for key in keys
        }
        # Detach mutable predictor settings; ensure this record is JSON serializable.
        effective = json.loads(json.dumps(effective, default=str, allow_nan=False))
        variant = {
            "requested_inference_confidence_threshold": self.inference_confidence_threshold,
            "post_filter_confidence_threshold": self.confidence_threshold,
            "effective_predictor_args": effective,
            "model_identity": dict(self._model_identity),
        }
        self._successful_predict_calls += 1
        if self._runtime_variants and self._runtime_variants[-1]["settings"] == variant:
            self._runtime_variants[-1]["predict_calls"] += 1
        else:
            self._runtime_variants.append({"settings": variant, "predict_calls": 1})
        if effective is not None and effective.get("conf") is not None:
            if float(effective["conf"]) != self.inference_confidence_threshold:
                raise RuntimeError("YOLO effective inference confidence differs from requested configuration")

    def get_runtime_audit(self) -> Dict[str, Any]:
        """Snapshot for runner history and standalone reports; survives GPU release.

        Only calls since begin_runtime_audit() are counted. Cached reference
        detections and metadata-only evaluations do not count as new predictions.
        """
        return copy.deepcopy({
            "schema_version": "driveloop_yolo_runtime.v1",
            "scope": "successful_predict_calls_since_begin_runtime_audit",
            "status": "recorded" if self._successful_predict_calls else "not_invoked",
            "configured_inference_confidence_threshold": self.inference_confidence_threshold,
            "configured_post_filter_confidence_threshold": self.confidence_threshold,
            "successful_predict_calls": self._successful_predict_calls,
            "variants": self._runtime_variants,
        })

    def _load_model(self) -> Any:
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise RuntimeError("ultralytics is required for YOLO detection") from exc
        return YOLO(self.weights)

    def release_gpu(self) -> None:
        """Drop the model and free the CUDA cache; the model is reloaded
        lazily on the next detect call.

        The experiment driver holds this detector while the DD2
        generation subprocess needs nearly the whole card (measured
        2026-07-09: detector-held 1.5 GiB caused CUDA OOM in the UNet
        after several closed-loop iterations on a 22 GiB A10). Moving
        the weights to CPU is NOT sufficient: the ultralytics predictor
        caches its device and then feeds CUDA inputs to CPU weights
        (measured crash on the following evaluation)."""
        self.model = None
        try:
            import gc

            import torch
        except ImportError:
            return
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def detect(self, frame: Any, frame_index: int) -> List[Detection]:
        if self.model is None:
            self.model = self._load_model()
            self._snapshot_model_identity()
        source = frame
        temp_path: Path | None = None
        if not isinstance(frame, (str, Path)):
            try:
                import cv2  # type: ignore
                temp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                temp_path = Path(temp.name)
                temp.close()
                cv2.imwrite(str(temp_path), frame)
                source = str(temp_path)
            except Exception as exc:
                raise RuntimeError("could not materialize frame for YOLO detector") from exc

        try:
            results = self.model.predict(
                source=source, verbose=False, conf=self.inference_confidence_threshold,
            )
            self._record_runtime()
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink()
                except FileNotFoundError:
                    pass

        if not results:
            return []
        result = results[0]
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        detections: List[Detection] = []
        for box in boxes:
            confidence = float(box.conf[0]) if getattr(box, "conf", None) is not None else 0.0
            if confidence < self.confidence_threshold:
                continue
            cls_id = int(box.cls[0]) if getattr(box, "cls", None) is not None else -1
            xyxy = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    frame_index=frame_index,
                    label=str(names.get(cls_id, cls_id)),
                    confidence=confidence,
                    box=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                )
            )
        return detections


class SimpleIoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_frame_gap: int = 1) -> None:
        self.iou_threshold = iou_threshold
        self.max_frame_gap = max_frame_gap
        self.next_track_id = 1
        self.tracks: List[Track] = []

    def update(self, detections: Iterable[Detection]) -> None:
        assigned: set[int] = set()
        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            best_track = None
            best_iou = 0.0
            for track in self.tracks:
                if track.track_id in assigned:
                    continue
                if track.label != detection.label:
                    continue
                if detection.frame_index - track.last_frame > self.max_frame_gap:
                    continue
                iou = detection.iou(track.detections[-1])
                if iou > best_iou:
                    best_iou = iou
                    best_track = track

            if best_track is not None and best_iou >= self.iou_threshold:
                best_track.append(detection)
                assigned.add(best_track.track_id)
            else:
                track = Track(self.next_track_id, detection.label, [detection])
                self.tracks.append(track)
                assigned.add(track.track_id)
                self.next_track_id += 1


@dataclass(frozen=True)
class PerceptionVideoWeights:
    coverage: float = 0.3
    confidence: float = 0.2
    tracking: float = 0.2
    identity: float = 0.2
    box_stability: float = 0.1

    def normalized(self) -> "PerceptionVideoWeights":
        total = self.coverage + self.confidence + self.tracking + self.identity + self.box_stability
        if total <= 0:
            return self
        return PerceptionVideoWeights(
            coverage=self.coverage / total,
            confidence=self.confidence / total,
            tracking=self.tracking / total,
            identity=self.identity / total,
            box_stability=self.box_stability / total,
        )


class PerceptionVideoEvaluator(BaseEvaluator):
    DETECTION_METADATA_KEYS = ("perception_detections", "detections_by_frame", "video_detections")

    def __init__(
        self,
        detector: VideoDetector | None = None,
        frame_reader: VideoFrameReader | None = None,
        target_labels: Iterable[str] | None = None,
        weights: PerceptionVideoWeights | None = None,
        confidence_threshold: float = 0.25,
        pass_threshold: float = 0.8,
        max_frames: int | None = None,
        tracker_iou_threshold: float = 0.3,
        static_motion_threshold: float = 0.5,
        min_target_support_frames: int = 2,
    ) -> None:
        self.detector = detector
        self.frame_reader = frame_reader or OpenCVFrameReader()
        self.target_labels = {self._normalize_label(label) for label in target_labels or []}
        self.weights = (weights or PerceptionVideoWeights()).normalized()
        self.confidence_threshold = confidence_threshold
        self.pass_threshold = pass_threshold
        self.max_frames = max_frames
        self.tracker_iou_threshold = tracker_iou_threshold
        self.static_motion_threshold = static_motion_threshold
        self.min_target_support_frames = int(min_target_support_frames)

    def evaluate(self, generation: Generation) -> Evaluation:
        detections, frame_count, measured, setup_reasons = self._collect_detections(generation)
        target_labels = self._resolve_target_labels(generation)
        metrics: Dict[str, float] = {
            "perception_measured": 1.0 if measured else 0.0,
            "perception_frame_count": float(frame_count),
        }
        reasons = list(setup_reasons)
        actions: List[str] = []

        if not measured:
            actions.append("provide YOLO weights, a detector backend, or precomputed detection JSON")
            return Evaluation(0.0, metrics, Diagnosis(False, reasons, actions))

        if frame_count <= 0:
            return Evaluation(0.0, metrics, Diagnosis(False, ["video_has_no_readable_frames"], ["verify video decoding"]))

        filtered = [
            d for d in detections
            if d.confidence >= self.confidence_threshold
            and (not target_labels or self._normalize_label(d.label) in target_labels)
        ]

        tracks = self._build_tracks(filtered)

        q_cov = self._coverage_score(filtered, frame_count)
        q_conf = self._confidence_score(filtered)
        q_track, dominant = self._track_score(tracks, frame_count)
        q_id = self._identity_score(filtered, dominant)
        q_box = self._box_stability_score(dominant)
        support_frames = len({d.frame_index for d in filtered})
        support_guard_triggered = 0 < support_frames < self.min_target_support_frames
        if support_guard_triggered:
            # Integrity guard: single-frame target support degenerates
            # Q_id/Q_box to 1.0 (2026-07-08 m5 class-flip forensics);
            # deny that free credit.
            q_id = 0.0
            q_box = 0.0
        score = round(
            self.weights.coverage * q_cov
            + self.weights.confidence * q_conf
            + self.weights.tracking * q_track
            + self.weights.identity * q_id
            + self.weights.box_stability * q_box,
            6,
        )

        motion_px, motion_norm = self._dominant_motion(dominant)
        metrics.update({
            "Q_cov": q_cov,
            "Q_conf": q_conf,
            "Q_track": q_track,
            "Q_id": q_id,
            "Q_box": q_box,
            "perception_detection_count": float(len(filtered)),
            "perception_track_count": float(len(tracks)),
            "perception_target_support_frames": float(support_frames),
            "perception_dominant_track_length": float(len(dominant.detections) if dominant else 0),
            "perception_dominant_net_motion_px": float(motion_px) if motion_px is not None else -1.0,
            "perception_dominant_motion_over_width": float(motion_norm) if motion_norm is not None else -1.0,
        })

        if support_guard_triggered:
            reasons.append("insufficient_target_support_frames")
            actions.append(
                "target must be detected in at least %d frames"
                % self.min_target_support_frames
            )
        if not filtered:
            reasons.append("target_object_not_detected")
            actions.append("make the target actor visible or check detector classes")
        if q_cov < self.pass_threshold:
            reasons.append("low_detection_coverage")
            actions.append("make the target actor visible across more frames")
        if q_conf < self.pass_threshold:
            reasons.append("low_detector_confidence")
            actions.append("increase object clarity, lighting, or scale")
        if q_track < self.pass_threshold:
            reasons.append("unstable_track_coverage")
            actions.append("reduce occlusion and keep motion temporally coherent")
        if q_id < self.pass_threshold:
            reasons.append("identity_inconsistent")
            actions.append("reduce identity switches")
        if q_box < self.pass_threshold:
            reasons.append("unstable_bounding_boxes")
            actions.append("stabilize target object box position and scale")
        request_implies_motion = bool(
            (generation.metadata.get("scene_specification") or {}).get("motion_primitives")
        )
        if (
            request_implies_motion
            and dominant is not None
            and len(dominant.detections) >= 3
            and motion_norm is not None
            and motion_norm < self.static_motion_threshold
        ):
            reasons.append("target_appears_static")
            actions.append(
                "increase visible target motion: strengthen maneuver wording or bind a source scene with existing target motion"
            )

        passed = score >= self.pass_threshold and not reasons
        return Evaluation(score, metrics, Diagnosis(passed, list(dict.fromkeys(reasons)), list(dict.fromkeys(actions))))

    def build_report(self, generation: Generation) -> Dict[str, Any]:
        reset_audit = getattr(self.detector, "begin_runtime_audit", None)
        if callable(reset_audit):
            reset_audit()
        evaluation = self.evaluate(generation)
        audit = getattr(self.detector, "get_runtime_audit", None)
        runtime = {"detector_runtime": audit()} if callable(audit) else {}
        measured = evaluation.metrics.get("perception_measured") == 1.0
        claim = "not_measured" if not measured else ("measured_passed" if evaluation.diagnosis.passed else "measured_failed")
        return {
            "schema_version": "driveloop_perception_video_eval.v0",
            **runtime,
            "generation": {"iteration": generation.iteration, "prompt": generation.prompt, "artifacts": dict(generation.artifacts)},
            "evaluation": asdict(evaluation),
            "interpretation": {
                "perception_claim": claim,
                "semantic_success_claim": "not_proven_by_perception_metrics_alone",
                "claim_boundary": "This report measures detector/tracker evidence for Eq. (15); it does not prove full prompt-video semantic success by itself.",
            },
        }

    def _collect_detections(self, generation: Generation) -> tuple[List[Detection], int, bool, List[str]]:
        metadata_detections = self._detections_from_metadata(generation.metadata)
        if metadata_detections is not None:
            return metadata_detections, self._frame_count_from_metadata(generation.metadata, metadata_detections), True, []

        video_path = generation.artifacts.get("video")
        if not video_path:
            return [], 0, False, ["missing_generation_artifact"]
        if self.detector is None:
            return [], 0, False, ["perception_detector_not_configured"]

        frames = self.frame_reader.read(Path(video_path), max_frames=self.max_frames)
        detections: List[Detection] = []
        for index, frame in enumerate(frames):
            detections.extend(self.detector.detect(frame, index))
        return detections, len(frames), True, []

    def _detections_from_metadata(self, metadata: Dict[str, Any]) -> List[Detection] | None:
        payload = None
        for key in self.DETECTION_METADATA_KEYS:
            if key in metadata:
                payload = metadata[key]
                break
        if payload is None:
            return None

        if isinstance(payload, dict) and isinstance(payload.get("frames"), list):
            detections: List[Detection] = []
            for frame in payload["frames"]:
                if not isinstance(frame, dict):
                    continue
                frame_index = int(frame.get("frame_index", 0))
                for item in frame.get("detections", []):
                    if isinstance(item, dict):
                        detections.append(Detection.from_dict(item, frame_index=frame_index))
            return detections

        if isinstance(payload, dict):
            detections = []
            for key, items in payload.items():
                if key in ("frame_count", "schema_version"):
                    continue
                if not isinstance(items, list):
                    continue
                for item in items:
                    if isinstance(item, dict):
                        detections.append(Detection.from_dict(item, frame_index=int(key)))
            return detections

        if isinstance(payload, list):
            return [Detection.from_dict(item) for item in payload if isinstance(item, dict)]
        return None

    def _frame_count_from_metadata(self, metadata: Dict[str, Any], detections: List[Detection]) -> int:
        for key in self.DETECTION_METADATA_KEYS:
            payload = metadata.get(key)
            if isinstance(payload, dict) and payload.get("frame_count") is not None:
                return int(payload["frame_count"])
        if metadata.get("frame_count") is not None:
            return int(metadata["frame_count"])
        return max((d.frame_index for d in detections), default=-1) + 1

    def _resolve_target_labels(self, generation: Generation) -> set[str]:
        labels = set(self.target_labels)
        metadata_labels = generation.metadata.get("target_labels")
        if isinstance(metadata_labels, list):
            labels.update(self._normalize_label(str(label)) for label in metadata_labels)
        import re
        # Strip ego-references so "toward the ego vehicle" does not make the
        # ego car a detection target (target-label leakage fix).
        prompt = re.sub(r"ego[\s-]+(vehicle|car|lane|path)", " ", generation.prompt.lower())
        for label in ("motorcycle", "car", "truck", "bus", "pedestrian", "bicycle"):
            if label in prompt:
                labels.add(label)
        return labels

    def _build_tracks(self, detections: List[Detection]) -> List[Track]:
        """Group by detector-provided track ids (e.g. BoT-SORT); fall back to IoU."""
        if any(d.track_id is not None for d in detections):
            grouped: Dict[Any, Track] = {}
            next_synthetic = -1
            for detection in sorted(detections, key=lambda d: d.frame_index):
                key = detection.track_id
                if key is None:
                    key = next_synthetic
                    next_synthetic -= 1
                track = grouped.get((key, detection.label))
                if track is None:
                    track = Track(int(key) if isinstance(key, int) else 0, detection.label, [])
                    grouped[(key, detection.label)] = track
                track.append(detection)
            return list(grouped.values())
        tracker = SimpleIoUTracker(iou_threshold=self.tracker_iou_threshold)
        for frame_index in sorted({d.frame_index for d in detections}):
            tracker.update(d for d in detections if d.frame_index == frame_index)
        return tracker.tracks

    def _dominant_motion(self, dominant: Track | None):
        """Net centroid displacement of the dominant track, normalized by mean box width."""
        if dominant is None or len(dominant.detections) < 2:
            return None, None
        centroids = [
            ((d.box[0] + d.box[2]) / 2.0, (d.box[1] + d.box[3]) / 2.0)
            for d in dominant.detections
        ]
        net = (
            (centroids[-1][0] - centroids[0][0]) ** 2
            + (centroids[-1][1] - centroids[0][1]) ** 2
        ) ** 0.5
        mean_width = sum(max(d.box[2] - d.box[0], 1e-6) for d in dominant.detections) / len(dominant.detections)
        return round(net, 3), round(net / mean_width, 6)

    def _coverage_score(self, detections: List[Detection], frame_count: int) -> float:
        return round(len({d.frame_index for d in detections}) / frame_count, 6) if frame_count > 0 else 0.0

    def _confidence_score(self, detections: List[Detection]) -> float:
        return round(sum(d.confidence for d in detections) / len(detections), 6) if detections else 0.0

    def _track_score(self, tracks: List[Track], frame_count: int) -> tuple[float, Track | None]:
        if not tracks or frame_count <= 0:
            return 0.0, None
        dominant = max(tracks, key=lambda track: len(track.detections))
        return round(len(dominant.detections) / frame_count, 6), dominant

    def _identity_score(self, detections: List[Detection], dominant: Track | None) -> float:
        return round(len(dominant.detections) / len(detections), 6) if detections and dominant else 0.0

    def _box_stability_score(self, dominant: Track | None) -> float:
        if dominant is None or not dominant.detections:
            return 0.0
        if len(dominant.detections) == 1:
            return 1.0
        ious = [cur.iou(prev) for prev, cur in zip(dominant.detections, dominant.detections[1:])]
        return round(sum(ious) / len(ious), 6) if ious else 0.0

    def _normalize_label(self, label: str) -> str:
        aliases = {"motorbike": "motorcycle", "bike": "bicycle", "cyclist": "bicycle", "person": "pedestrian", "vehicle": "car", "vehicles": "car"}
        return aliases.get(label.lower().strip().replace("-", "_"), label.lower().strip().replace("-", "_"))
