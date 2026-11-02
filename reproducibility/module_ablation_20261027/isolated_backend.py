"""Post-hoc intervention, isolated from frozen DD2 code and evidence.

Disable all synthetic target box injection, including static boxes and fallback,
from candidate zero. Retain compatible identity-bound real-track reinforcement.
This is broader than disabling the refiner's escalation flag alone.
"""
import copy
import os
from driveloop.backends.drivedreamer2 import DriveDreamer2Backend

BACKEND_VERSION = 'without_synthetic_injection_v2'


def suppress_synthetic(override):
    result = copy.deepcopy(override)
    if result.get('available') is not True:
        raise ValueError('Cannot verify intervention without an available override')
    boxes = result['boxes3d']
    mapping = result['audit']['actor_motion_frame_mapping']
    counts = {}
    retained_real = []
    for surface in ('append', 'per_frame_append', 'per_frame_append_ego'):
        entries = boxes[surface]
        if not isinstance(entries, list):
            raise ValueError('Unknown box surface: '+surface)
        counts[surface] = 0
        for entry in entries:
            if entry.get('category') not in ('bus', 'truck'):
                raise ValueError('Intervention restricted to bus/truck protocol')
            if entry.get('source') == 'source_bound_real_track':
                if not (surface == 'per_frame_append_ego'
                        and entry.get('provenance') == 'driveloop_real_track_ego_injection'
                        and entry.get('synthetic_track_id') is None
                        and mapping.get('mode') == 'source_bound_real_track_ego'
                        and mapping.get('available') is True
                        and mapping.get('selection_basis') == 'binding_instance_token'
                        and mapping.get('trajectory_compatibility', {}).get('compatible') is True):
                    raise ValueError('Unverified real-track entry')
                retained_real.append(entry)
            else:
                expected = ('class_default_dimensions', 'driveloop_executable_condition') if surface == 'append' else (
                    'actor_motion_plan.per_frame_actor_boxes3d', 'driveloop_actor_motion_surface')
                if (entry.get('source'), entry.get('provenance')) != expected:
                    raise ValueError('Unknown injection provenance on '+surface)
                if surface == 'per_frame_append_ego' and mapping.get('mode') != 'ego_frame_one_entry_per_video_frame':
                    raise ValueError('Unexpected synthetic mapping mode')
                counts[surface] += 1
    if retained_real and counts['per_frame_append_ego']:
        raise ValueError('Mixed real/synthetic ego entries need a separate protocol')
    boxes.update(append=[], per_frame_append=[], per_frame_append_ego=retained_real,
                 mode='append_and_per_frame_append_ego' if retained_real else 'append')
    result['audit']['synthetic_intervention'] = {
        'version': BACKEND_VERSION, 'removed_counts': counts,
        'retained_real_count': len(retained_real),
        'pre_intervention_mapping': copy.deepcopy(mapping),
        'initial_candidate_included': True,
        'native_dataset_boxes': 'untouched; append-only override lists filtered',
    }
    if not retained_real:
        result['audit']['actor_motion_frame_mapping'] = {
            'available': False, 'reason': 'synthetic_injection_suppressed_v2',
            'mapped_entry_count': 0,
        }
    result['audit']['control_level'] = 'tensor_override_runtime' if retained_real else 'runtime_surface_observation'
    result['audit']['limitations'] = [
        'real_track_reinforcement_only' if retained_real else 'no_added_target_boxes',
        'source_scene_and_text_conditions_retained',
        'input_intervention_not_video_semantic_proof',
    ]
    return result


class NoSyntheticBackend(DriveDreamer2Backend):
    def _build_override_json(self, *args, **kwargs):
        if os.environ.get('DRIVELOOP_EGO_INJECTION') != '1':
            raise ValueError('V2 protocol requires the frozen ego-injection path')
        if os.environ.get('DRIVELOOP_EGO_REQUIRE_REAL_TRACK') == '1':
            raise ValueError('Conflicting real-track-only flag')
        # The reference checks remain intact. Even a forced-synthetic request
        # must try the real track; synthetic fallback is filtered before loading.
        old = getattr(self, '_synthetic_trajectory_escalation', False)
        try:
            self._synthetic_trajectory_escalation = False
            candidate = super()._build_override_json(*args, **kwargs)
        finally:
            self._synthetic_trajectory_escalation = old
        return suppress_synthetic(candidate)
