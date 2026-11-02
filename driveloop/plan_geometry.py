"""Fail-closed checks for source-conditioned diagnostic plans, before generation.

Uses known source footprints, not generated-video semantics or a vehicle model.
The caller must supply all relevant annotations and conservative swept footprints.
"""
from __future__ import annotations
import math


def check_plan_geometry(footprints, source_lane, target_lane, obstacles, *,
                        swept_pairs=None, minimum_clearance=.5, minimum_lane_fraction=.95):
    if not math.isfinite(minimum_clearance) or minimum_clearance < 0:
        raise ValueError('invalid minimum clearance')
    if not math.isfinite(minimum_lane_fraction) or not 0 < minimum_lane_fraction <= 1:
        raise ValueError('invalid lane fraction')
    reasons=[]
    n=len(footprints)
    if n<2 or len(obstacles)!=n:
        return {'geometry_preflight_passed':False,'reasons':['missing_or_misaligned_frames']}
    polygons=[source_lane,target_lane,*footprints]
    polygons.extend(p for frame in obstacles for p in frame)
    if any(p is None or p.is_empty or not p.is_valid or p.area<=0 for p in polygons):
        return {'geometry_preflight_passed':False,'reasons':['invalid_geometry']}
    if any(not frame for frame in obstacles):reasons.append('missing_source_obstacle_evidence')
    start_fraction=footprints[0].intersection(source_lane).area/footprints[0].area
    end_fraction=footprints[-1].intersection(target_lane).area/footprints[-1].area
    if start_fraction+1e-9<minimum_lane_fraction:reasons.append('start_outside_source_lane')
    if end_fraction+1e-9<minimum_lane_fraction:reasons.append('incomplete_target_lane_entry')
    distances=[min((p.distance(o) for o in frame),default=None) for p,frame in zip(footprints,obstacles)]
    collision_frames=[i for i,(p,frame) in enumerate(zip(footprints,obstacles)) if any(p.intersection(o).area>1e-9 for o in frame)]
    if collision_frames:reasons.append('source_footprint_overlap')
    if any(d is not None and d+1e-9<minimum_clearance for d in distances):reasons.append('insufficient_source_clearance')
    swept_distances=[]
    if swept_pairs is None or len(swept_pairs)!=n-1:
        reasons.append('missing_interval_evidence')
    else:
        for p,frame in swept_pairs:
            if p is None or p.is_empty or not p.is_valid or p.area<=0 or not frame or any(o is None or o.is_empty or not o.is_valid or o.area<=0 for o in frame):
                reasons.append('invalid_interval_evidence');swept_distances.append(None);continue
            d=min(p.distance(o) for o in frame);swept_distances.append(d)
            if any(p.intersection(o).area>1e-9 for o in frame) or d+1e-9<minimum_clearance:
                reasons.append('interval_overlap_or_insufficient_clearance')
    return {'geometry_preflight_passed':not reasons,'reasons':list(dict.fromkeys(reasons)),
            'start_lane_fraction':start_fraction,'end_lane_fraction':end_fraction,
            'frame_clearances_m':distances,'collision_frames':collision_frames,
            'interval_clearances_m':swept_distances,'minimum_clearance_m':minimum_clearance,
            'minimum_lane_fraction':minimum_lane_fraction,'vehicle_dynamics_validated':False,
            'video_semantic_success':False}


def require_geometry_ready(audit):
    if audit.get('geometry_preflight_passed') is not True:
        raise ValueError('plan geometry rejected: '+', '.join(audit.get('reasons', ['missing audit'])))
