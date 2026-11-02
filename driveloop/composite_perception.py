from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from driveloop.perception_video import PerceptionVideoEvaluator
from driveloop.schema import Diagnosis, Evaluation, Generation


@dataclass(frozen=True)
class CompositeVideoLayout:
    """Layout of the DD2 tester debug mosaic.

    The DD2 tester saves a composite video: source frames and condition
    visualization rows are stacked above the generated row, and camera views
    are tiled horizontally. Perception must only see the generated row,
    split into per-camera views.
    """

    view_width: int = 448
    generated_row_height: int = 256
    num_views: int = 6
    generated_row_at_bottom: bool = True
    # Mosaic tile order of the DD2 tester composite video. Must match the
    # dataloader camera order; verify before trusting target-view scores.
    cam_order: tuple = (
        "cam_front_left",
        "cam_front",
        "cam_front_right",
        "cam_back_right",
        "cam_back",
        "cam_back_left",
    )

    def extract_view(self, frame: Any, view_index: int) -> Any:
        height = frame.shape[0]
        y0 = height - self.generated_row_height if self.generated_row_at_bottom else 0
        x0 = view_index * self.view_width
        return frame[y0:y0 + self.generated_row_height, x0:x0 + self.view_width]

    def matches(self, frame: Any) -> bool:
        height, width = frame.shape[:2]
        return width >= self.view_width * self.num_views and height > self.generated_row_height


class CompositePerceptionVideoEvaluator(PerceptionVideoEvaluator):
    """Perception evaluator that crops the generated row of a DD2 composite
    video and evaluates each camera view independently, reporting the best
    view. Falls back to whole-frame evaluation for non-composite videos."""

    def __init__(
        self,
        *args: Any,
        layout: CompositeVideoLayout | None = None,
        baseline_video: Any = None,
        baseline_iou_threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.layout = layout or CompositeVideoLayout()
        self.baseline_video = baseline_video
        self.baseline_iou_threshold = float(baseline_iou_threshold)
        self._baseline_cache: Dict[str, dict] = {}

    def resolve_baseline_video(self, generation: Generation) -> str | None:
        """The no-injection support video actually used to subtract pre-existing
        source objects from this generation. Request metadata wins over the
        constructor default. Single source of truth, so the runner can archive
        which support video a run really used."""
        path = generation.metadata.get("perception_baseline_video") or self.baseline_video
        return str(path) if path else None

    def evaluate(self, generation: Generation) -> Evaluation:
        # Reset on every call, including early returns, to prevent stale tracks.
        self._view_detection_frames = {}
        if self._detections_from_metadata(generation.metadata) is not None:
            return super().evaluate(generation)
        video_path = generation.artifacts.get("video")
        if not video_path or self.detector is None:
            return super().evaluate(generation)

        frames = self.frame_reader.read(Path(video_path), max_frames=self.max_frames)
        if not frames:
            return Evaluation(
                0.0,
                {"perception_measured": 1.0, "perception_frame_count": 0.0},
                Diagnosis(False, ["video_has_no_readable_frames"], ["verify video decoding"]),
            )
        if not self.layout.matches(frames[0]):
            return super().evaluate(generation)

        baseline_path = self.resolve_baseline_video(generation)
        baseline_table = (
            self._baseline_view_detections(baseline_path)
            if baseline_path and Path(str(baseline_path)).exists()
            else {}
        )
        baseline_removed_total = 0

        best_evaluation: Evaluation | None = None
        best_view = -1
        view_scores: Dict[int, float] = {}
        view_evaluations: Dict[int, Evaluation] = {}
        for view_index in self._views_to_evaluate(generation.metadata):
            reset = getattr(self.detector, "reset", None)
            if callable(reset):
                reset()
            payload_frames = []
            brightness_sum = 0.0
            view_centers: list = []
            for frame_index, frame in enumerate(frames):
                view = self.layout.extract_view(frame, view_index)
                brightness_sum += float(view.mean())
                detections = self.detector.detect(view, frame_index)
                if baseline_table:
                    detections, removed = self._subtract_baseline(
                        detections, baseline_table.get((view_index, frame_index), [])
                    )
                    baseline_removed_total += removed
                if detections:
                    view_centers.append(
                        [
                            (str(d.label).lower(), (d.box[0] + d.box[2]) / 2.0)
                            for d in detections
                        ]
                    )
                else:
                    view_centers.append(None)
                payload_frames.append({
                    "frame_index": frame_index,
                    "detections": [
                        {
                            "frame_index": d.frame_index,
                            "label": d.label,
                            "confidence": d.confidence,
                            "box": list(d.box),
                            "track_id": d.track_id,
                        }
                        for d in detections
                    ],
                })
            view_generation = Generation(
                iteration=generation.iteration,
                prompt=generation.prompt,
                artifacts=dict(generation.artifacts),
                metadata={
                    **generation.metadata,
                    "perception_detections": {"frames": payload_frames, "frame_count": len(frames)},
                },
            )
            evaluation = super().evaluate(view_generation)
            self._view_detection_frames[view_index] = payload_frames
            view_scores[view_index] = evaluation.score
            view_evaluations[view_index] = evaluation
            if not hasattr(self, "_view_centers"):
                self._view_centers = {}
            self._view_centers[view_index] = view_centers
            view_brightness = round(brightness_sum / max(len(frames), 1), 3)
            if not hasattr(self, "_view_brightness"):
                self._view_brightness = {}
            self._view_brightness[view_index] = view_brightness
            if best_evaluation is None or evaluation.score > best_evaluation.score:
                best_evaluation = evaluation
                best_view = view_index

        target_view_indices = self._target_view_indices(generation.metadata)
        selected_evaluation = best_evaluation
        selected_view = best_view
        if target_view_indices:
            in_target = [vi for vi in target_view_indices if vi in view_evaluations]
            if in_target:
                selected_view = max(in_target, key=lambda vi: view_evaluations[vi].score)
                selected_evaluation = view_evaluations[selected_view]

        metrics = dict(selected_evaluation.metrics)
        for view_index, score in view_scores.items():
            metrics["perception_view%d_score" % view_index] = score
        metrics["perception_best_view"] = float(best_view)
        metrics["perception_all_view_max_score"] = best_evaluation.score
        metrics["perception_selected_view"] = float(selected_view)
        metrics["perception_target_view_count"] = float(len(target_view_indices))
        metrics["perception_baseline_available"] = 1.0 if baseline_table else 0.0
        metrics["perception_baseline_subtracted_count"] = float(baseline_removed_total)

        diagnosis = selected_evaluation.diagnosis
        direction = self._maneuver_direction_check(
            generation.metadata,
            getattr(self, "_view_centers", {}).get(selected_view, []),
        )
        if direction is not None:
            expected_sign, pixel_delta, consistent = direction
            metrics["maneuver_expected_pixel_sign"] = expected_sign
            metrics["maneuver_pixel_delta_x"] = round(pixel_delta, 3)
            metrics["maneuver_direction_consistent"] = 1.0 if consistent else 0.0
            if not consistent:
                from driveloop.schema import Diagnosis
                diagnosis = Diagnosis(
                    False,
                    list(diagnosis.reasons) + ["maneuver_direction_mismatch"],
                    list(diagnosis.suggested_actions)
                    + ["verify signed lateral geometry and source-scene distractors"],
                )
        return Evaluation(selected_evaluation.score, metrics, diagnosis)

    def _direction_label_set(self, category: str) -> set:
        """Label vocabulary for the maneuver-direction probe, aligned
        with the scoring vocabulary: normalized labels (the raw
        detector says person, the scorer says pedestrian), widened by
        the same super-class expansion the subclass scorer uses. The
        raw-label exact match this replaces silently made direction
        unmeasurable for pedestrian targets and stricter than the
        score itself (2026-07-22 direction-densification record)."""
        return {self._normalize_label(str(category))} if category else set()

    def _maneuver_direction_check(self, metadata, centers):
        """Expected pixel motion of the injected actor in the selected
        view: approaching the ego lane means moving toward the image
        center, i.e. pixel x increases for side=-1 (left) and decreases
        for side=+1 (right). Returns (expected_sign, delta, consistent)
        or None when not measurable."""
        if not isinstance(metadata, dict):
            return None
        plan = metadata.get("dd2_override_candidate_plan")
        surface = plan.get("actor_motion_surface_plan") if isinstance(plan, dict) else None
        if not isinstance(surface, dict) or surface.get("maneuver") not in (
            "cut_in",
            "lane_change",
        ):
            return None
        side = float(surface.get("lateral_side") or 0.0)
        if side == 0.0:
            return None
        target_actor = surface.get("target_actor") or {}
        category = str(target_actor.get("category") or "").lower()
        allowed = self._direction_label_set(category)
        observed = []
        for frame_dets in centers:
            if not frame_dets:
                continue
            values = [
                center
                for label, center in frame_dets
                if not allowed or self._normalize_label(str(label)) in allowed
            ]
            if values:
                observed.append(sum(values) / len(values))
        if len(observed) < 3:
            return None
        expected_sign = 1.0 if side < 0 else -1.0
        pixel_delta = observed[-1] - observed[0]
        consistent = pixel_delta * expected_sign > 0
        return expected_sign, pixel_delta, consistent
        brightness_map = getattr(self, "_view_brightness", {})
        for view_index, value in brightness_map.items():
            metrics["perception_view%d_brightness" % view_index] = value
        if best_view in brightness_map:
            metrics["perception_best_view_brightness"] = brightness_map[best_view]
        metrics["perception_layout_views"] = float(self.layout.num_views)
        metrics["perception_generated_row_height"] = float(self.layout.generated_row_height)
    def _baseline_view_detections(self, path: Any) -> dict:
        """Detections of the no-injection baseline video of the same
        source window, keyed by (view_index, frame_index). Determinism
        makes the baseline exact; class-agnostic IoU subtraction removes
        pre-existing source objects from target credit (2026-07-08
        baseline-differential record)."""
        key = str(path)
        if key not in self._baseline_cache:
            table: dict = {}
            frames = self.frame_reader.read(Path(key), max_frames=self.max_frames)
            if frames and self.layout.matches(frames[0]) and self.detector is not None:
                for view_index in range(self.layout.num_views):
                    reset = getattr(self.detector, "reset", None)
                    if callable(reset):
                        reset()
                    for frame_index, frame in enumerate(frames):
                        view = self.layout.extract_view(frame, view_index)
                        table[(view_index, frame_index)] = self.detector.detect(view, frame_index)
            self._baseline_cache[key] = table
        return self._baseline_cache[key]

    @staticmethod
    def _iou_xyxy(a, b) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        union = (
            max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
            + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            - inter
        )
        return inter / union if union > 0 else 0.0

    def _subtract_baseline(self, detections, baseline_dets):
        if not baseline_dets:
            return detections, 0
        kept, removed = [], 0
        for d in detections:
            if any(
                self._iou_xyxy(d.box, b.box) >= self.baseline_iou_threshold
                for b in baseline_dets
            ):
                removed += 1
            else:
                kept.append(d)
        return kept, removed

    def _views_to_evaluate(self, metadata: dict) -> list:
        """View indices to evaluate; the v9 default scores every mosaic view."""
        return list(range(self.layout.num_views))

    def _target_view_indices(self, metadata: dict) -> list:
        """Resolve target cam types from the generation metadata to mosaic
        view indices. Empty result preserves legacy all-view-max behavior."""
        if not isinstance(metadata, dict):
            return []
        plan = metadata.get("dd2_override_candidate_plan")
        surface = plan.get("actor_motion_surface_plan") if isinstance(plan, dict) else None
        cams = surface.get("target_cam_types") if isinstance(surface, dict) else None
        if not cams:
            return []
        indices = []
        for cam in cams:
            cam_l = str(cam).lower()
            if cam_l in self.layout.cam_order:
                indices.append(self.layout.cam_order.index(cam_l))
        return sorted(set(indices))
