"""Source/proposal/contract checks only; no detector, loader or diffusion."""
import json
from pathlib import Path
from types import SimpleNamespace
from driveloop.native_phase import NativePhaseRegistry, digest
from driveloop.dynamic_phase import PhaseAdmission, action_id
from driveloop.dynamic_phase_pilot import AmplitudePilotRefiner, METHODS
from driveloop.phase_amplitude import propose_phase

ROOT=Path('/home/lesliton/projects/DriveLoop')
OUT=ROOT/'outputs/revision_20261016_candidate5614_plan'
PRE=ROOT/'outputs/revision_20261015_candidate5614_inputs/preflight.json'
PRE_SHA='a82687280eaf6639c00f9b44ad79f519077a8d6e8c9740e9a372eb3b9528a550'
PLAN=ROOT/'outputs/revision_20261015_bounded_source_plan/protocol.json'
TRACK=ROOT/'outputs/revision_20261015_source_expansion/selected_candidates.pkl'

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    assert digest(PLAN)=='a16756d5cc9c64cb32c0a075f1ab7e564c9504b128ca4407870b9c0a1a69ab92'
    assert digest(TRACK)=='7a1f2110751aa19078bfb525540528461175810b7c627abc40eb968cb9769a78'
    plan=json.loads(PLAN.read_text())
    assert plan['eligible_for_runtime_admission']==[5614]
    record=next(x for x in plan['candidate_records'] if x['candidate_index']==5614)
    assert record['grid_interval_s']==[-.375,.5] and record['fixed_three_strategy_grid_supported']
    files=[PLAN,TRACK,Path(__file__).resolve()]
    curve=[]
    for row in record['geometry_grid']:
        assert row['passed'] and digest(row['check_path'])==row['check_sha256']
        assert digest(row['manifest'])==row['manifest_sha256']
        files.extend([Path(row['check_path']),Path(row['manifest'])])
        curve.append({'phase_s':row['phase_s'],'source_mean_dx_px':row['source_mean_dx_px'],'geometry_passed':True})
    assert [x['phase_s'] for x in curve]==[-.375,-.25,-.125,0,.125,.25,.375,.5]
    assert all(a['source_mean_dx_px']>b['source_mean_dx_px'] for a,b in zip(curve,curve[1:]))
    r=NativePhaseRegistry(PRE,PRE_SHA)
    assert r.candidate==5614 and r.scene==record['scene_token']
    for direction,phase in [('delayed',.25),('advanced',-.25)]:
        assert next(x for x in curve if x['phase_s']==phase)['source_mean_dx_px']==r.means[direction]
    (OUT/'results.json').write_text(json.dumps({'rows':curve,'source_only':True,'generation_run':False,'bounded_plan_sha256':digest(PLAN)},indent=2)+'\n')
    def reject_unexpected_admission(*args):
        raise AssertionError('Policy check must not invoke actual input admission')
    admission=PhaseAdmission(r,curve,reject_unexpected_admission,OUT/'synthetic_policy_checks')
    checks=[]
    for phase in [-.375,-.25,-.125,0,.125,.25,.375,.5]:
        for observed in [-1e6,1e6]:
            proposal=propose_phase(curve,phase,0,[observed]*4)
            assert -.375<=proposal['proposed_phase_s']<=.5
            assert abs(proposal['proposed_phase_s']-phase)<=.125+1e-12
            assert not proposal['runtime_authorized']
        checks.append('bounds_and_step_'+str(phase))
    for phase in [-.5,.50001]:
        try:admission.contract(admission.request(0,phase=phase))
        except ValueError:pass
        else:raise AssertionError('Out-of-range phase accepted')
        checks.append('reject_outside_'+str(phase))
    for phase in [-.25,.25]:
        target=next(x['source_mean_dx_px'] for x in curve if x['phase_s']==phase)
        for method in METHODS:
            refiner=AmplitudePilotRefiner(admission,method,phase)
            request=admission.request(target,phase=phase)
            sequence=[phase]
            for iteration in [0,1]:
                q=admission.contract(request)
                feedback={'contract':q,'iteration':iteration,'late_deltas':[target]*4,'status':'measured','error_px':0}
                result=refiner.refine(request,SimpleNamespace(metrics={'native_phase_feedback':feedback}))
                # All policies must preserve target and registry identity.
                request=admission.request(target,phase=result.condition['native_phase']['phase_s'])
                assert result.condition['native_phase']==admission.contract(request)
                sequence.append(request.condition['native_phase']['phase_s'])
            sign=1 if phase>0 else -1
            expected=[phase,sign*.125,sign*.375] if method=='fixed_amplitude_search' else [phase]*3
            assert sequence==expected,(method,sequence)
            checks.append(method+'_'+str(phase))
        held=propose_phase(curve,phase,target,[None,target,target,target])
        assert held['status']=='hold_unmeasured' and held['proposed_phase_s']==phase
        checks.append('missing_measurement_'+str(phase))
    assert admission.events==[] and admission.receipts=={}
    r.verify()
    files.append(OUT/'results.json')
    files.extend(ROOT/'driveloop'/name for name in ('dynamic_phase.py','dynamic_phase_pilot.py','phase_amplitude.py','native_phase.py'))
    result={'passed':True,'generation_run':False,'actual_loader_calls':0,'diffusion_calls':0,'checks':checks,'fixed_hashes':{str(p):digest(p) for p in files},'registry_sha256':r.fingerprint,'scope':'Synthetic policy and frozen source geometry checks; not actual DD2 admission for additional phases or video effect.'}
    (OUT/'policy_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print('CANDIDATE5614_POLICY_CHECK_PASS',len(checks),flush=True)

if __name__=='__main__':main()
