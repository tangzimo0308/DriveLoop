"""Versioned direction correction; S_perc, detector, baseline and views stay v10b."""
from driveloop.perception_v10 import ManeuverViewRestrictedSuperclassEvaluator
from driveloop.schema import Evaluation, Diagnosis
from driveloop.track_direction import track_direction, PROTOCOL


class ContinuousTrackPerceptionEvaluator(ManeuverViewRestrictedSuperclassEvaluator):
    def _maneuver_direction_check(self, metadata, centers):
        # Never allow the legacy per-frame class-average direction into v11.
        return None

    def evaluate(self, generation):
        evaluation = super().evaluate(generation)
        metrics = dict(evaluation.metrics)
        for key in ('maneuver_direction_consistent', 'maneuver_expected_pixel_sign', 'maneuver_pixel_delta_x'):
            metrics.pop(key, None)
        selected = int(metrics.get('perception_selected_view', -1))
        frames = getattr(self, '_view_detection_frames', {}).get(selected)
        surface = (generation.metadata.get('dd2_override_candidate_plan') or {}).get('actor_motion_surface_plan') or {}
        reason = None
        if frames is None: reason = 'missing_per_frame_boxes'
        elif selected != self.layout.cam_order.index('cam_front'): reason = 'unsupported_direction_view'
        elif surface.get('maneuver') not in ('cut_in', 'lane_change'): reason = 'unsupported_maneuver'
        category = (surface.get('target_actor') or {}).get('category')
        if not category: reason = 'missing_target_category'
        if reason:
            audit = {'protocol': PROTOCOL, 'measured': False, 'consistent': None, 'reason': reason, 'tracks': []}
        else:
            allowed = self._direction_label_set(str(category))
            filtered = [[{**d, 'label': self._normalize_label(str(d['label']))} for d in frame['detections']
                         if d['confidence'] >= self.confidence_threshold
                         and self._normalize_label(str(d['label'])) in allowed] for frame in frames]
            audit = track_direction(filtered, self._lateral_side(generation.metadata))
        self.last_direction_audit = audit
        metrics.update(perception_scorer_version=11.0, maneuver_direction_measured=float(audit['measured']),
                       maneuver_direction_track_count=float(len(audit['tracks'])),
                       maneuver_direction_support_frames=float(audit.get('support', 0)))
        reasons = list(evaluation.diagnosis.reasons)
        actions = list(evaluation.diagnosis.suggested_actions)
        if audit['measured']:
            metrics.update(maneuver_direction_consistent=float(audit['consistent']),
                           maneuver_expected_pixel_sign=audit['expected_sign'], maneuver_pixel_delta_x=audit['pixel_delta_x'])
            if not audit['consistent']:
                reasons.append('maneuver_direction_mismatch')
        else:
            reasons.append('maneuver_direction_unmeasured_' + audit['reason'])
            actions.append('verify one unambiguous target track; missing direction cannot pass the joint gate')
        passed = evaluation.diagnosis.passed and audit['measured'] and audit['consistent']
        return Evaluation(evaluation.score, metrics, Diagnosis(bool(passed), list(dict.fromkeys(reasons)), actions))
