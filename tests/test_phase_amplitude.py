import pytest
from driveloop.phase_amplitude import propose_phase

CURVE=[{'phase_s':i/8,'source_mean_dx_px':-50*i/8,'geometry_passed':True} for i in range(-4,5)]
@pytest.mark.parametrize('residual,expected',[(4,-.08),(-4,.08),(8,-.125),(-8,.125)])
def test_residual_magnitude_and_trust_bound(residual,expected):
 p=propose_phase(CURVE,0,residual,[0]*4)
 assert p['proposed_phase_s']==pytest.approx(expected)
 assert abs(p['step_s'])<=.125 and not p['runtime_authorized']
 assert p['saturated']==(abs(residual)>6.25)

def test_missing_not_zero():
 p=propose_phase(CURVE,0,4,[0,None,0,0]);assert p['status']=='hold_unmeasured' and p['residual_px'] is None

def test_boundary_clamp():
 p=propose_phase(CURVE,.5,-40,[-25]*4);assert p['proposed_phase_s']==.5 and p['saturated']

@pytest.mark.parametrize('change',[{'geometry_passed':False},{'source_mean_dx_px':100},{'phase_s':-.5}])
def test_reject_holes_nonmonotone_or_duplicate(change):
 c=[dict(p) for p in CURVE];c[3].update(change)
 with pytest.raises(ValueError):propose_phase(c,0,4,[0]*4)

@pytest.mark.parametrize('value',[float('nan'),float('inf'),True])
def test_invalid_measurement(value):
 with pytest.raises(ValueError):propose_phase(CURVE,0,4,[0,value,0,0])

def test_no_extrapolation():
 with pytest.raises(ValueError):propose_phase(CURVE,.6,4,[0]*4)

def test_small_error_hold():
 assert propose_phase(CURVE,0,.5,[0]*4)['status']=='hold_within_software_deadband'

def test_increasing_curve_supported():
 c=[{**p,'source_mean_dx_px':-p['source_mean_dx_px']} for p in CURVE]
 assert propose_phase(c,0,4,[0]*4)['proposed_phase_s']==pytest.approx(.08)
