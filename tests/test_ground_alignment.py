import copy
import numpy as np
import pytest
from driveloop.ground_alignment import align_to_local_vehicle_bottom

T=[[0,0,1,1.5],[-1,0,0,0],[0,-1,0,1.5],[0,0,0,1]]

def inputs(height=3.2):
    entries=[{'relative_frame_idx':0,'source':'synthetic','actor_id':'bus','ego':{'center_ego':[20,3,-0.3],'dims':[2.6,height,10], 'heading_ego':0.1}}]
    record={'relative_step':0,'record':{'boxes3d':np.array([[-3,0.75,19,2,1.5,4,0,0,0],[-2,0.75,21,2,1.5,4,0,0,0]]), 'ori_labels3d':['vehicle.car']*2, 'calib':{'cam2ego':T}}}
    return entries,[record]

@pytest.mark.parametrize('height',[1.5,3.2,4.8])
def test_places_bottom_on_ground_without_changing_other_controls(height):
    entries,records=inputs(height);original=copy.deepcopy(entries)
    out,audit=align_to_local_vehicle_bottom(entries,records)
    assert entries==original
    assert out[0]['ego']['center_ego']==[20,3,height/2]
    assert out[0]['ego']['dims']==entries[0]['ego']['dims']
    assert out[0]['ego']['heading_ego']==0.1
    assert audit['frames'][0]['estimated_ground_z_m']==0

@pytest.mark.parametrize('case',['missing','far','spread','nan'])
def test_unverified_ground_fails_before_generation(case):
    entries,records=inputs()
    if case=='missing':records=[]
    elif case=='far':records[0]['record']['boxes3d'][:,2]=100
    elif case=='spread':records[0]['record']['boxes3d'][1,1]=3
    else:records[0]['record']['boxes3d'][1,1]=float('nan')
    with pytest.raises(ValueError):align_to_local_vehicle_bottom(entries,records)

def test_real_track_is_not_moved():
    entries,_=inputs();entries[0]['source']='source_bound_real_track'
    out,audit=align_to_local_vehicle_bottom(entries,[])
    assert out==entries and not audit['frames']
