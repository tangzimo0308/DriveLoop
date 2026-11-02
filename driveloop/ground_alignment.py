"""Opt-in geometric diagnostic: estimate local road height from annotated vehicles.

This is a local, flat-road approximation, not a measured road surface. No video
scores enter the estimate. Ambiguous/missing support fails before generation.
"""
import copy
import numpy as np
from driveloop.ego_injection import cam_box9_to_ego_entry

MODE = 'local_vehicle_bottom_v1'
ALLOWED_LABELS = {'vehicle.car', 'vehicle.truck', 'vehicle.bus.rigid', 'vehicle.bus.bendy'}


def align_to_local_vehicle_bottom(entries, front_records):
    records = {int(item['relative_step']): item['record'] for item in front_records}
    output = copy.deepcopy(entries)
    audits = []
    for entry in output:
        if entry.get('source') == 'source_bound_real_track':
            continue
        step = int(entry['relative_frame_idx'])
        if step not in records:
            raise ValueError(f'ground alignment: missing source frame {step}')
        record = records[step]
        center = np.asarray(entry['ego']['center_ego'], dtype=float)
        height = float(entry['ego']['dims'][1])
        if not np.isfinite(center).all() or not np.isfinite(height) or height <= 0:
            raise ValueError('ground alignment: invalid synthetic geometry')
        support = []
        for index, (box, label) in enumerate(zip(record.get('boxes3d', []), record.get('ori_labels3d', []))):
            if label not in ALLOWED_LABELS:
                continue
            box = np.asarray(box, dtype=float)
            if box.shape != (9,) or not np.isfinite(box).all() or np.any(box[3:6] <= 0):
                continue
            ref = cam_box9_to_ego_entry(box, record['calib']['cam2ego'])
            distance = float(np.linalg.norm(np.asarray(ref['center_ego'][:2]) - center[:2]))
            if distance <= 15.0:
                support.append({'source_box_index': index, 'label': label, 'distance_xy_m': distance,
                                'bottom_z_m': ref['center_ego'][2] - ref['dims'][1] / 2})
        support = sorted(support, key=lambda item: (item['distance_xy_m'], item['source_box_index']))[:3]
        if len(support) < 2:
            raise ValueError(f'ground alignment: fewer than two local vehicle supports at frame {step}')
        bottoms = [item['bottom_z_m'] for item in support]
        if not np.isfinite(bottoms).all() or np.ptp(bottoms) > 0.75:
            raise ValueError(f'ground alignment: inconsistent local support heights at frame {step}')
        ground = float(np.median(bottoms))
        new_z = ground + height / 2
        old_z = float(center[2])
        # Preserve x/y, dimensions, orientation, identity, and all other controls.
        entry['ego']['center_ego'][2] = new_z
        evidence = {'relative_frame_idx': step, 'old_center_z_m': old_z, 'new_center_z_m': new_z,
                    'old_bottom_z_m': old_z - height/2, 'estimated_ground_z_m': ground,
                    'support': support}
        entry['ground_alignment'] = evidence
        audits.append(evidence)
    return output, {'mode': MODE, 'frames': audits,
                    'maximum_support_distance_m': 15.0, 'maximum_support_spread_m': 0.75,
                    'claim_boundary': 'Local flat-road estimate from annotated vehicle bottoms; not measured terrain or video success.'}
