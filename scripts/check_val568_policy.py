"""Frozen source and policy checks; no detector, model loader or generation."""
import json, math
from pathlib import Path
from types import SimpleNamespace
from driveloop.native_phase import NativePhaseRegistry, digest
from driveloop.dynamic_phase import PhaseAdmission
from driveloop.dynamic_phase_pilot import AmplitudePilotRefiner, METHODS
from driveloop.phase_amplitude import propose_phase

ROOT=Path('/home/lesliton/projects/DriveLoop')
OUT=ROOT/'outputs/revision_20261019_val568_plan'
PRE=ROOT/'outputs/revision_20261019_val_inputs/val_candidate568/preflight.json'
PRE_SHA='efcba203506046a5fb04def05b6cd1a1eda3e83916d418f933dd31a876112f12'
PLAN=ROOT/'outputs/revision_20261019_measurable_source_plan/protocol.json'
TRACK=ROOT/'outputs/revision_20261019_val_source_projection/selected_candidates.pkl'
GEOM=ROOT/'outputs/revision_20261019_val_source_geometry_v2/results.json'
MEAS=ROOT/'outputs/revision_20261019_val_source_measurability/results.json'

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    assert digest(PLAN)=='8388cc8af4aee8aaba2912e0e1a4fa9f5173fa42a05dc0ec1e517646fe8e35ef'
    assert digest(TRACK)=='fc6557582ebbf4c155ec9236d1c3a80f58d09f2cd3c0f0cdc707aca41aa53935'
    assert digest(GEOM)=='a814659ea5a285f64e458dc6f3dc93a15ac3d0f23639d1d9695ad9d7381f9b84'
    plan=json.loads(PLAN.read_text());geo=json.loads(GEOM.read_text())
    r=NativePhaseRegistry(PRE,PRE_SHA)
    assert r.candidate==568 and r.scene not in plan['excluded_scenes']
    record=next(x for x in geo['rows'] if x['candidate_index']==568)
    measured=next(x for x in json.loads(MEAS.read_text())['rows'] if x['candidate_index']==568)
    assert measured['primary_pass'] and all(measured['valid_frames']['fasterrcnn'][4:8])
    assert record['scene_token']==measured['scene_token']==r.scene
    assert record['instance_token']==measured['instance_token']==r.instance
    assert record['grid_bounds_s']==[-.5,.5] and record['eligible_for_runtime']
    files=[PLAN,TRACK,GEOM,MEAS,Path(__file__).resolve()]
    curve=[]
    for row in record['phases']:
        assert row['passed'] and digest(row['check_path'])==row['check_sha256']
        assert digest(row['manifest'])==row['manifest_sha256']
        files.extend([Path(row['check_path']),Path(row['manifest'])])
        curve.append({'phase_s':row['phase_s'],'source_mean_dx_px':row['source_mean_dx_px'],'geometry_passed':True})
    assert [x['phase_s'] for x in curve]==[-.5,-.375,-.25,-.125,0,.125,.25,.375,.5]
    assert all(a['source_mean_dx_px']<b['source_mean_dx_px'] for a,b in zip(curve,curve[1:]))
    for direction,phase in [('delayed',.25),('advanced',-.25)]:
        assert math.isclose(next(x['source_mean_dx_px'] for x in curve if x['phase_s']==phase),r.means[direction],abs_tol=1e-10,rel_tol=0)
    assert r.means['delayed']>0>r.means['advanced']
    (OUT/'results.json').write_text(json.dumps({'rows':curve,'source_only':True,'generation_run':False,'source_plan_sha256':digest(PLAN)},indent=2)+'\n')
    def forbidden(*args):raise AssertionError('Policy check must not invoke actual input admission')
    admission=PhaseAdmission(r,curve,forbidden,OUT/'synthetic_policy_checks')
    checks=[]
    for phase in [x['phase_s'] for x in curve]:
        for observed in [-1e6,1e6]:
            proposal=propose_phase(curve,phase,0,[observed]*4)
            assert -.5<=proposal['proposed_phase_s']<=.5
            assert abs(proposal['proposed_phase_s']-phase)<=.125+1e-12
            assert not proposal['runtime_authorized']
        checks.append('bounds_and_step_'+str(phase))
    for phase in [-.50001,.50001]:
        try:admission.contract(admission.request(0,phase=phase))
        except ValueError:pass
        else:raise AssertionError('Out-of-range phase accepted')
        checks.append('reject_outside_'+str(phase))
    for phase in [-.25,.25]:
        target=next(x['source_mean_dx_px'] for x in curve if x['phase_s']==phase)
        for method in METHODS:
            refiner=AmplitudePilotRefiner(admission,method,phase)
            request=admission.request(target,phase=phase);sequence=[phase]
            for iteration in [0,1]:
                q=admission.contract(request)
                f={'contract':q,'iteration':iteration,'late_deltas':[target]*4,'status':'measured','error_px':0}
                result=refiner.refine(request,SimpleNamespace(metrics={'native_phase_feedback':f}))
                request=admission.request(target,phase=result.condition['native_phase']['phase_s'])
                assert result.condition['native_phase']==admission.contract(request)
                sequence.append(request.condition['native_phase']['phase_s'])
            sign=1 if phase>0 else -1
            assert sequence==([phase,sign*.125,sign*.375] if method=='fixed_amplitude_search' else [phase]*3)
            checks.append(method+'_'+str(phase))
        held=propose_phase(curve,phase,target,[None,target,target,target])
        assert held['status']=='hold_unmeasured' and held['proposed_phase_s']==phase
        checks.append('missing_measurement_'+str(phase))
        # This source curve increases with phase. Positive residual must increase
        # phase even for the negative (advanced) initial target, and vice versa.
        for residual in [-5,5]:
            q=admission.contract(admission.request(target,phase=phase))
            f={'contract':q,'iteration':0,'late_deltas':[target-residual]*4,'status':'measured','error_px':residual}
            refiner=AmplitudePilotRefiner(admission,'residual_feedback',phase)
            revised=refiner.refine(admission.request(target,phase=phase),SimpleNamespace(metrics={'native_phase_feedback':f}))
            delta=revised.condition['native_phase']['phase_s']-phase
            assert delta*residual>0 and abs(delta)<=.125+1e-12
            assert revised.condition['native_phase']['target_dx_px']==target
            checks.append('increasing_curve_residual_'+str(phase)+'_'+str(residual))
    assert admission.events==[] and admission.receipts=={}
    r.verify();files.append(OUT/'results.json')
    files.extend(ROOT/'driveloop'/name for name in ('dynamic_phase.py','dynamic_phase_pilot.py','phase_amplitude.py','native_phase.py'))
    result={'passed':True,'generation_run':False,'actual_loader_calls':0,'diffusion_calls':0,'checks':checks,'fixed_hashes':{str(p):digest(p) for p in files},'registry_sha256':r.fingerprint,'scope':'Synthetic policies and frozen source geometry only. Other exact phases still require DD2 admission; generated measurability and effectiveness unverified.'}
    (OUT/'policy_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print('VAL568_POLICY_CHECK_PASS',len(checks),flush=True)

if __name__=='__main__':main()
