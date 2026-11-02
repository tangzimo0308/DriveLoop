from track_audit import audit_tracks

def box(x):return {'box':[x,0,x+10,10],'label':'bus','score':0.9}

def test_contiguous_positive_motion():
    t=audit_tracks([[box(0)],[box(1)],[box(2)]])['tracks']
    assert len(t)==1 and t[0]['positive_pixel_direction'] and t[0]['pixel_delta_x']==2

def test_missing_frame_breaks_track():
    t=audit_tracks([[box(0)],[],[box(1)]])['tracks']
    assert len(t)==2 and not any(x['direction_measurable'] for x in t)

def test_average_of_different_actors_is_not_a_track():
    t=audit_tracks([[box(0)],[box(0),box(100)],[box(100)]])['tracks']
    assert len(t)==2 and not any(x['direction_measurable'] for x in t)

def test_duplicate_boxes_not_assigned_twice():
    r=audit_tracks([[box(0)],[box(0),box(1)],[box(1)]])
    assert len(r['tracks'])==2 and r['ambiguous_transition_frames']==[1,2]
    assert sum(t['support'] for t in r['tracks'])==4

def test_empty_predictions():
    assert audit_tracks([[],[],[]])['tracks']==[]
