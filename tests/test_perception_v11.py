import copy
import numpy as np
import pytest
from driveloop.track_direction import track_direction
from driveloop.perception_v11 import ContinuousTrackPerceptionEvaluator
from driveloop.perception_v10 import ManeuverViewRestrictedSuperclassEvaluator
from driveloop.perception_video import Detection
from driveloop.schema import Generation
from driveloop.acceptance import joint_gate


def box(x,label='bus'):return {'box':[x,10,x+20,30],'label':label,'confidence':0.9}

@pytest.mark.parametrize('side,xs,expected',[(-1,[0,1,2],True),(1,[2,1,0],True),(-1,[2,1,0],False),(-1,[2,2,2],False)])
def test_single_track_direction(side,xs,expected):
    a=track_direction([[box(x)] for x in xs],side)
    assert a['measured'] and a['consistent']==expected

@pytest.mark.parametrize('frames,reason',[
    ([[box(0)],[],[box(1)]],'insufficient_consecutive_support'),
    ([[box(0)],[box(0),box(100)],[box(100)]],'insufficient_consecutive_support'),
    ([[box(0)],[box(0),box(1)],[box(1)]],'ambiguous_target_association'),
    ([[box(0),box(100)]]*3,'multiple_candidate_tracks'),
    ([[box(0)],[box(1,'truck')],[box(2)]],'insufficient_consecutive_support'),
    ([[box(float('nan'))]]*3,'invalid_detection_geometry')])
def test_uncertain_cases_are_unmeasured(frames,reason):
    a=track_direction(frames,-1)
    assert not a['measured'] and a['consistent'] is None and a['reason']==reason

class Reader:
    def __init__(self,count):self.count=count
    def read(self,*args,**kwargs):
        f=np.zeros((784,2688,3),dtype=np.uint8);f[-256:,448:896]=255
        return [f]*self.count

class Detector:
    def __init__(self,frames):self.frames=frames
    def detect(self,frame,index):
        if frame.mean()==0:return []
        return [Detection(index,d['label'],d['confidence'],tuple(d['box'])) for d in self.frames[index]]

def generation():
    return Generation(0,'bus cuts in from left',artifacts={'video':'fake.mp4'},metadata={'dd2_override_candidate_plan':{'actor_motion_surface_plan':{
        'target_actor':{'category':'bus'},'target_cam_types':['cam_front'],'maneuver':'cut_in','lateral_side':-1}}})

def test_integration_removes_false_mean_direction_and_preserves_score():
    frames=[[box(0)],[box(0),box(100)],[box(100)]]
    kwargs={'frame_reader':Reader(3),'detector':Detector(frames),'target_labels':['bus'],'confidence_threshold':0.2}
    old=ManeuverViewRestrictedSuperclassEvaluator(**kwargs).evaluate(generation())
    new=ContinuousTrackPerceptionEvaluator(**kwargs).evaluate(generation())
    assert old.metrics['maneuver_direction_consistent']==1
    assert new.score==old.score
    assert 'maneuver_direction_consistent' not in new.metrics
    assert joint_gate(new.metrics,.3,raw_score=new.score)=='UNMEASURED'

def test_reuse_does_not_keep_previous_video_tracks():
    frames=[[box(0)],[box(1)],[box(2)]]
    ev=ContinuousTrackPerceptionEvaluator(frame_reader=Reader(3),detector=Detector(frames),target_labels=['bus'])
    assert ev.evaluate(generation()).metrics['maneuver_direction_consistent']==1
    g=generation();g.artifacts.clear()
    assert 'maneuver_direction_consistent' not in ev.evaluate(g).metrics
    assert ev._view_detection_frames=={}
