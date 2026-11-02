import pytest
from shapely.geometry import box
from driveloop.plan_geometry import check_plan_geometry,require_geometry_ready


def valid():
    return dict(footprints=[box(0,0,2,2),box(3,0,5,2)],source_lane=box(-1,-1,2.5,3),
                target_lane=box(2.5,-1,6,3),obstacles=[[box(10,0,11,1)],[box(10,0,11,1)]],
                swept_pairs=[(box(0,0,5,2),[box(10,0,11,1)])])

def test_clear_transition_passes_without_claiming_video_or_dynamics():
    a=check_plan_geometry(**valid());require_geometry_ready(a)
    assert a['geometry_preflight_passed'] and not a['video_semantic_success'] and not a['vehicle_dynamics_validated']

def test_center_entry_is_not_enough_for_long_vehicle():
    args=valid();args['footprints'][-1]=box(2,0,5,2)
    a=check_plan_geometry(**args);assert 'incomplete_target_lane_entry' in a['reasons']

def test_small_obstacles_are_not_filtered_by_class_or_area():
    args=valid();args['obstacles'][1].append(box(4,1,4.1,1.1))
    a=check_plan_geometry(**args);assert a['collision_frames']==[1]
    with pytest.raises(ValueError,match='rejected'):require_geometry_ready(a)

def test_interval_obstacle_rejects_even_when_samples_are_clear():
    args=valid();args['swept_pairs']=[(box(0,0,5,2),[box(2.4,0,2.6,1)])]
    a=check_plan_geometry(**args);assert 'interval_overlap_or_insufficient_clearance' in a['reasons']

@pytest.mark.parametrize('field,value,reason',[
    ('obstacles',[],'missing_or_misaligned_frames'),
    ('obstacles',[[],[]],'missing_source_obstacle_evidence'),
    ('swept_pairs',None,'missing_interval_evidence'),
    ('source_lane',box(0,0,0,0),'invalid_geometry'),
])
def test_missing_evidence_fails_closed(field,value,reason):
    args=valid();args[field]=value
    assert reason in check_plan_geometry(**args)['reasons']

def test_clearance_threshold_boundary():
    args=valid();args['obstacles']=[[box(2.5,0,3,1)],[box(5.5,0,6,1)]]
    assert check_plan_geometry(**args)['geometry_preflight_passed']
    args['obstacles'][1]=[box(5.49,0,6,1)]
    assert not check_plan_geometry(**args)['geometry_preflight_passed']

def test_missing_audit_cannot_authorize_generation():
    with pytest.raises(ValueError):require_geometry_ready({})
