"""Diagnostic association only; consecutive IoU links do not prove identity."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a,b):
    area=lambda c:max(0.,c[2]-c[0])*max(0.,c[3]-c[1])
    inter=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    union=area(a)+area(b)-inter
    return inter/union if union>0 else 0.


def audit_tracks(frames,threshold=0.3):
    tracks=[];ambiguities=[]
    for step,detections in enumerate(frames):
        active=[t for t in tracks if t['detections'][-1]['frame']==step-1]
        matched=set()
        if active and detections:
            matrix=np.array([[iou(t['detections'][-1]['box'],d['box']) for d in detections] for t in active])
            if np.any((matrix>=threshold).sum(axis=0)>1) or np.any((matrix>=threshold).sum(axis=1)>1):
                ambiguities.append(step)
            rows,cols=linear_sum_assignment(np.where(matrix>=threshold,1-matrix,1e6))
            for row,col in zip(rows,cols):
                if matrix[row,col]<threshold:continue
                active[row]['detections'].append({**detections[col],'frame':step,'link_iou':float(matrix[row,col])})
                matched.add(int(col))
        for index,detection in enumerate(detections):
            if index not in matched:tracks.append({'id':len(tracks),'detections':[{**detection,'frame':step}]})
    for track in tracks:
        ds=track['detections'];centers=[(d['box'][0]+d['box'][2])/2 for d in ds]
        track.update(frames=[d['frame'] for d in ds],support=len(ds),first_center_x=centers[0],last_center_x=centers[-1],
                     pixel_delta_x=centers[-1]-centers[0],direction_measurable=len(ds)>=3,
                     positive_pixel_direction=None if len(ds)<3 else centers[-1]>centers[0],
                     association_ambiguous=any(d['frame'] in ambiguities for d in ds))
    return {'tracks':tracks,'ambiguous_transition_frames':ambiguities,'threshold':threshold,'max_missing_frames':0,
            'claim':'One-to-one consecutive-frame geometric association only; no appearance identity, ego-motion compensation, or lane-entry proof.'}
