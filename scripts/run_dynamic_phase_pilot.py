"""User-run development comparison. New exact parameters are checked before generation."""
import argparse,json,pickle,time,traceback
from pathlib import Path
from dataclasses import asdict
from driveloop.dynamic_phase import PhaseAdmission,DynamicPhaseBackend,DynamicPhaseRunner
from driveloop.dynamic_phase_pilot import METHODS,AmplitudePilotRefiner
from driveloop.native_phase import NativePhaseRegistry,digest
from driveloop.native_phase_png import PNGPhaseMeasurement,DETECTOR_SHA
from driveloop.native_phase_online import ReleasingDetector,OnlineEvaluator
from driveloop.schema import DriveLoopConfig
from scripts.run_native_bus_control import ROOT,setup_env,make_backend
from scripts.geometry_with_source_support import check_geometry
from scripts.validate_phase_parameter import validate_phase_parameter

PRE=ROOT/'outputs/revision_20261008_candidate1583_inputs/preflight.json'
PRE_SHA='997384cdb119b1679abc84e48276da54f38e3141d297271e50e192bd499d9af2'
RANGE=ROOT/'outputs/revision_20261013_phase_range'
GATE=ROOT/'outputs/revision_20261014_amplitude_pilot_launchcheck/check.json'
DETECTOR='/data/models/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth'

def save(path,obj):Path(path).write_text(json.dumps(obj,indent=2,default=str)+'\n')
def specification(r):
    files=list((ROOT/'driveloop').rglob('*.py'))
    files += [ROOT/'scripts'/name for name in ('run_dynamic_phase_pilot.py','run_dynamic_phase_pilot.sh','validate_phase_parameter.py','geometry_with_source_support.py','run_native_bus_control.py','dd2_capture_python','dd2_capture_hooks.py','run_native_matched_capture.py','preflight_candidate1583.py')]
    return {'version':'amplitude_development_pilot_v1','registry_sha256':r.fingerprint,'methods':list(METHODS),
        'targets_px':{'delayed':r.means['delayed'],'advanced':r.means['advanced']},'initial_phase_s':{'delayed':.25,'advanced':-.25},
        'seed_offsets':[0,1,2],'max_candidates_per_method':3,'reference_cost_per_method':3,'logical_render_cap_per_method':6,
        'actual_shared_reference_renders':3,'max_actual_candidates':18,'max_actual_new_renders':21,'max_logical_renders_across_six_runs':36,
        'reference_reuse':'Fresh original references shared across all six method runs; each run charged all three. Candidate generations not reused.',
        'fixed_search':'Initial target phase, then same sign 0.125s, then same sign 0.375s; no measured values used.',
        'feedback':'Starts from target phase; source inverse plus measured residual; max step 0.125s, software deadband1px, range[-0.5,0.5].',
        'selection':'Best complete four-frame absolute pixel error; missing values remain missing; no paper success rate.',
        'failure_policy':'Missing measurement stops that method; exact admission or integrity failure stops whole run; no automatic fallback or retry.',
        'stop_on_acceptance':False,'budget_scope':'Same generation-count caps, not equal total compute. Admission loader calls and seconds recorded separately, including shared cache.',
        'limits':'Previously inspected development scene and seeds; design changed after prior pilot failure. Not confirmatory, no physical trajectory or general efficacy claim.',
        'code_sha256':{str(p):digest(p) for p in sorted(set(files))},'curve_sha256':digest(RANGE/'results.json'),'detector_sha256':digest(DETECTOR)}

def verify(spec,r):
    r.verify()
    for p,h in spec['code_sha256'].items():
        if digest(p)!=h:raise ValueError('Frozen code changed: '+p)
    if digest(RANGE/'results.json')!=spec['curve_sha256'] or digest(DETECTOR)!=spec['detector_sha256']:raise ValueError('Frozen curve or detector changed')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',required=True,type=Path);parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    try:
        setup_env();r=NativePhaseRegistry(PRE,PRE_SHA);spec=specification(r);verify(spec,r)
        assert spec['detector_sha256']==DETECTOR_SHA
        curve=json.loads((RANGE/'results.json').read_text())['rows'];assert len(curve)==9 and all(x['geometry_passed'] for x in curve)
        chain=ROOT/'outputs/revision_20261014_dynamic_phase_audit'
        proof=json.loads((chain/'summary.json').read_text());assert proof['passed'] and not proof['generation_run']
        chain_protocol=json.loads((chain/'protocol.json').read_text())
        for p,h in chain_protocol['frozen_files'].items():assert digest(p)==h,p
        save(out/'protocol.json',{'specification':spec,'check_only':args.check_only,'written_before_generation':True,'prior_chain_audit_sha256':digest(chain/'summary.json')})
        if args.check_only:
            save(out/'check.json',{'passed':True,'generation_run':False,'specification':spec,'prior_chain_loader_calls':6,'new_loader_calls':0})
            print('AMPLITUDE_PILOT_LAUNCHCHECK_PASS',out,flush=True);return
        gate=json.loads(GATE.read_text())
        if not gate['passed'] or gate['generation_run'] or gate['specification']!=spec:raise ValueError('Launch preflight absent or stale')
        trackpath=ROOT/'outputs/revision_20261008_signal_precheck/extended_tracks.pkl'
        expected=json.loads((ROOT/'outputs/revision_20261008_signal_precheck/results.json').read_text())['track_cache_sha256']
        assert digest(trackpath)==expected;track=pickle.loads(trackpath.read_bytes())[1583]
        sourcepath=ROOT/'outputs/revision_20261007_native_multiscene_source/candidate1583/original/manifest.json';source=json.loads(sourcepath.read_text())
        provenance=[Path(p) for p in spec['code_sha256']]+[RANGE/'results.json',trackpath,sourcepath,out/'protocol.json']
        def admit(phase,dest):
            verify(spec,r);dest.mkdir(parents=True,exist_ok=False)
            gm=check_geometry(source,dest/'geometry',phase,track)
            if gm is None:raise ValueError('Exact geometry rejected; evidence retained')
            return validate_phase_parameter(r,dest/'geometry/manifest.json',dest/'input_preflight',phase,provenance)
        admission=PhaseAdmission(r,curve,admit,out/'admission')
        def factory(manifest,dest,audit):
            verify(spec,r)
            b=make_backend(manifest,dest,audit_only=audit);b.python_executable=str(ROOT/'scripts/dd2_capture_python');return b
        references=DynamicPhaseBackend(admission,factory,out/'references',False);refs={}
        for seed in (0,1,2):
            refs[seed]=references.generate(admission.request(0),seed);print('AMPLITUDE_REFERENCE_CAPTURED',seed,flush=True)
        summaries=[];ledger=list(references.events)
        for direction,phase in [('delayed',.25),('advanced',-.25)]:
            for method in METHODS:
                dest=out/direction/method;backend=DynamicPhaseBackend(admission,factory,dest/'candidates',False)
                def current_clip(g):
                    clip=backend.clip_for(g)
                    a=backend.capture_rows[g.artifacts['native_phase_receipt']]['latent_audit']
                    b=references.capture_rows[refs[g.iteration].artifacts['native_phase_receipt']]['latent_audit']
                    for k in ('initial_noise','image_latent_before'):
                        if a[k]['sha256']!=b[k]['sha256']:raise ValueError('Same-seed latent reference mismatch')
                    return clip
                detector=ReleasingDetector(DETECTOR)
                measure=PNGPhaseMeasurement(current_clip,lambda g:references.clip_for(refs[g.iteration]),detector)
                evaluator=OnlineEvaluator(measure,detector);refiner=AmplitudePilotRefiner(admission,method,phase)
                runner=DynamicPhaseRunner(backend,evaluator,refiner,DriveLoopConfig(max_iterations=3,target_score=0,stop_on_acceptance=False,output_dir=dest/'history'))
                before=len(admission.events);method_start=time.monotonic()
                result=runner.run(admission.request(spec['targets_px'][direction],phase=phase));save(dest/'result.json',asdict(result))
                rows=[]
                for a in result.attempt_history:
                    f=a.evaluation.metrics['native_phase_feedback'];q=f['contract'];error=f['error_px']
                    rows.append({'iteration':a.iteration,'phase_s':q['phase_s'],'observed_dx_px':f['observed_dx_px'],'absolute_error_px':abs(error) if error is not None else None,'measured':error is not None,'video':a.generation.artifacts['video']})
                valid=[a for a in rows if a['measured']];new=admission.events[before:]
                row={'method':method,'direction':direction,'attempts':rows,'best_measured':min(valid,key=lambda a:a['absolute_error_px']) if valid else None,
                    'actual_candidate_renders':len(backend.events),'logical_reference_renders':3,'logical_total_renders':3+len(backend.events),
                    'complete_three_candidate_budget':len(rows)==3,'all_attempts_measurable':all(a['measured'] for a in rows),
                    'new_exact_admissions':len(new),'new_admission_loader_calls':sum(json.loads(Path(x['receipt']).read_text())['actual_loader_calls'] for x in new),
                    'new_admission_seconds':sum(x['seconds'] for x in new),'method_wall_seconds':time.monotonic()-method_start}
                summaries.append(row);ledger.extend(backend.events);save(out/'partial_summary.json',summaries);save(out/'generation_ledger.json',ledger)
                print('AMPLITUDE_METHOD_COMPLETE',direction,method,'best_error',row['best_measured']['absolute_error_px'] if valid else None,flush=True)
        verify(spec,r)
        assert len(ledger)==3+sum(x['actual_candidate_renders'] for x in summaries)<=21
        save(out/'summary.json',{'complete':True,'development_only':True,'methods':summaries,'actual_total_renders':len(ledger),
            'logical_total_renders':sum(x['logical_total_renders'] for x in summaries),'shared_reference_renders':3,
            'actual_admission_loader_calls':sum(x['new_admission_loader_calls'] for x in summaries),'admissions':admission.events,
            'all_methods_completed_budget':all(x['complete_three_candidate_budget'] for x in summaries),'all_attempts_measurable':all(x['all_attempts_measurable'] for x in summaries),
            'generation_ledger':ledger,'elapsed_seconds':time.monotonic()-started,'paper_success_rate_computed':False})
        print('DYNAMIC_AMPLITUDE_PILOT_COMPLETE',out,flush=True)
    except Exception as exc:
        save(out/'failure.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'no_automatic_retry':True,'evidence_retained':True})
        raise
if __name__=='__main__':main()
