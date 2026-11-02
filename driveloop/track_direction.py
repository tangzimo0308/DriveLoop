"""v11 diagnostic direction rule: unique, consecutive, same-class IoU links.

No appearance identity or lane topology is available. Ambiguity is unmeasured,
not failure or success. Fixed thresholds must not be tuned per returned video.
"""
import math

PROTOCOL = 'unique_consecutive_track_v1'
IOU_THRESHOLD = 0.3
MIN_FRAMES = 3


def iou(a, b):
    inter = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    area = lambda c: (c[2]-c[0])*(c[3]-c[1])
    union = area(a)+area(b)-inter
    return inter/union if union > 0 else 0.


def track_direction(frames, side):
    audit = {'protocol': PROTOCOL, 'iou_threshold': IOU_THRESHOLD,
             'minimum_consecutive_frames': MIN_FRAMES, 'max_missing_frames': 0,
             'tracks': [], 'ambiguous_frames': [], 'invalid_detection_count': 0,
             'measured': False, 'consistent': None,
             'claim_boundary': 'Geometric association only; not verified actor identity, ego-motion compensation, or lane-entry proof.'}
    if not isinstance(side, (int, float)) or not math.isfinite(side) or side == 0:
        return {**audit, 'reason': 'missing_requested_side'}
    tracks = audit['tracks']
    for frame_index, frame in enumerate(frames):
        detections = []
        for det in frame:
            try:
                box = [float(v) for v in det['box']]
                valid = len(box) == 4 and all(math.isfinite(v) for v in box) and box[2] > box[0] and box[3] > box[1]
                if not valid: raise ValueError('invalid box')
                detections.append({**det, 'box': box, 'frame': frame_index})
            except (KeyError, TypeError, ValueError):
                audit['invalid_detection_count'] += 1
        active = [t for t in tracks if t['detections'][-1]['frame'] == frame_index-1]
        edges = [(i,j) for i,t in enumerate(active) for j,d in enumerate(detections)
                 if t['detections'][-1].get('label') == d.get('label')
                 and iou(t['detections'][-1]['box'], d['box']) >= IOU_THRESHOLD]
        row_degree = {i:sum(x==i for x,y in edges) for i in range(len(active))}
        col_degree = {j:sum(y==j for x,y in edges) for j in range(len(detections))}
        if any(v>1 for v in row_degree.values()) or any(v>1 for v in col_degree.values()):
            audit['ambiguous_frames'].append(frame_index)
        matched = set()
        for i,j in edges:
            if row_degree[i] == 1 and col_degree[j] == 1:
                active[i]['detections'].append(detections[j]); matched.add(j)
        for j,det in enumerate(detections):
            if j not in matched: tracks.append({'id': len(tracks), 'detections': [det]})
    for track in tracks:
        ds = track['detections'];xs = [(d['box'][0]+d['box'][2])/2 for d in ds]
        track.update(frames=[d['frame'] for d in ds], support=len(ds), pixel_delta_x=xs[-1]-xs[0])
    candidates = [t for t in tracks if t['support'] >= MIN_FRAMES]
    audit['candidate_count'] = len(candidates)
    if audit['invalid_detection_count']:
        reason = 'invalid_detection_geometry'
    elif audit['ambiguous_frames']:
        reason = 'ambiguous_target_association'
    elif not candidates:
        reason = 'insufficient_consecutive_support'
    elif len(candidates) != 1:
        reason = 'multiple_candidate_tracks'
    else:
        chosen = candidates[0];expected_sign = 1.0 if side < 0 else -1.0
        audit.update(measured=True, selected_track_id=chosen['id'], support=chosen['support'],
                     expected_sign=expected_sign, pixel_delta_x=chosen['pixel_delta_x'],
                     consistent=chosen['pixel_delta_x']*expected_sign > 0)
        reason = 'unique_consecutive_track'
    audit['reason'] = reason
    return audit
