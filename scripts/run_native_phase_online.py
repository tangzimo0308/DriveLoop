"""User-run, fixed development pilot: four policies, two directions, three attempts."""
import argparse, copy, json, os, pickle, time, traceback
from dataclasses import asdict, replace
from pathlib import Path
from driveloop.native_phase import NativePhaseRegistry, NativePhaseBackend, NativePhaseRunner, digest
from driveloop.native_phase_png import PNGPhaseMeasurement
from driveloop.native_phase_online import METHODS, SCHEDULE, PilotRefiner, ReleasingDetector, OnlineEvaluator, make_clip, method_summary
from driveloop.schema import DriveLoopConfig
from scripts.run_native_bus_control import ROOT, setup_env, make_backend
from scripts.run_native_advance_pair import check_pair

PREFLIGHT=ROOT/'outputs/revision_20261013_online_pilot_preflight'
PRE=ROOT/'outputs/revision_20261008_candidate1583_inputs/preflight.json'
PRE_SHA='997384cdb119b1679abc84e48276da54f38e3141d297271e50e192bd499d9af2'
DETECTOR='/data/models/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth'

def save(path, obj):Path(path).write_text(json.dumps(obj,indent=2,default=str)+'\n')

def plans_for(registry):
    import numpy as np
    from dreamer_datasets import boxes3d_utils
    plans={}
    for arm in SCHEDULE:
        records=pickle.loads((Path(registry.case['inputs'][arm]['dataset'])/'labels/data.pkl').read_bytes());boxes=[]
        for i in range(8):
            r=records[24+i*3];j=r['instance_tokens'].index(registry.instance)
            cs=boxes3d_utils.boxes3d_to_corners3d(r['boxes3d'][j:j+1],rot_axis=1)
            if cs[...,2].min()<=.5:raise ValueError('Invalid projection')
            p=boxes3d_utils.corners3d_to_corners2d(cs,cam_intrinsic=r['calib']['cam_intrinsic'])[0].astype(np.float64)*np.array([448/1600,256/900])
            boxes.append(tuple(np.r_[p.min(0),p.max(0)].tolist()))
        plans[arm]=boxes
    return plans

def specification(registry):
    files=list((ROOT/'driveloop').rglob('*.py'))
    files += [ROOT/'scripts'/n for n in ('run_native_phase_online.py','run_native_phase_online.sh','run_native_bus_control.py','run_native_advance_pair.py','run_native_matched_capture.py','dd2_capture_python','dd2_capture_hooks.py')]
    return {'version':'native_phase_online_development_v1','registry_sha256':registry.fingerprint,
        'candidate_index':registry.candidate,'methods':list(METHODS),'targets':{a:registry.means[a] for a in ('delayed','advanced')},
        'targets_basis':'Frozen source geometry; this already inspected scene is development-only.',
        'iterations_and_seed_offsets':[0,1,2],'seed_bank':0,'max_candidate_calls_per_method':3,
        'reference_calls_per_method':3,'logical_render_cap_per_method':6,
        'actual_reference_calls_shared_across_eight_runs':3,'max_actual_candidate_calls':24,'max_actual_new_renders':27,
        'reference_reuse':'Three fresh originals shared across methods and directions; charge all three to EACH method run. Candidate generations are never reused.',
        'stop_on_acceptance':False,'stop_on_missing':'Stop that method, retain failure and unused budget; unequal realized counts forbid equal-spend efficacy claim.',
        'primary':'Continuous absolute error against target over complete frames4-7; report every missing/ambiguous result.',
        'selection':'Best measured absolute error within each method; no zero imputation; identical selection across methods.',
        'software_tolerance_px':1,'tolerance_scope':'Development controller deadband only; no paper success threshold or success rate.',
        'fixed_schedule':list(SCHEDULE),'direct_baseline':'Target recipe from first attempt, with identical three seed offsets.',
        'limits':'Single development scene; no independent confirmatory claim, maneuver/physical truth claim, or full paper utility.',
        'code_sha256':{str(p):digest(p) for p in sorted(set(files))},'detector_sha256':digest(DETECTOR)}

def verify(spec, registry):
    registry.verify()
    for p,h in spec['code_sha256'].items():
        if digest(p)!=h:raise ValueError('Code changed: '+p)
    if digest(DETECTOR)!=spec['detector_sha256']:raise ValueError('Detector changed')

class CaptureStore:
    def __init__(self,registry,plans,spec):
        self.registry,self.plans,self.spec=registry,plans,spec
        self.rows={};self.clips={};self.references={};self.events=[]
    def factory(self,manifest,dest,audit_only):
        backend=make_backend(manifest,dest,audit_only=audit_only)
        backend.python_executable=str(ROOT/'scripts/dd2_capture_python')
        owner=self
        arm=next(a for a in SCHEDULE if self.registry.case['inputs'][a]['dataset']==manifest['dataset'])
        class Capturing:
            def generate(self,request,iteration):
                verify(owner.spec,owner.registry)
                capture=dest/'capture';latent_path=dest/'latent_audit.json'
                names={'DRIVELOOP_CAPTURE_DIR':str(capture),'DRIVELOOP_DD2_IMAGE_CONDITION_AUDIT_PATH':str(latent_path)}
                previous={k:os.environ.get(k) for k in names};os.environ.update(names);start=time.monotonic()
                try:g=backend.generate(request,iteration)
                finally:
                    for k,v in previous.items():
                        if v is None:os.environ.pop(k,None)
                        else:os.environ[k]=v
                if audit_only:
                    h=json.loads((capture/'hook_state.json').read_text())
                    if h['counts']!={'decode':0,'postprocess':0,'composite':0} or latent_path.exists() or 'video' in g.artifacts:
                        raise ValueError('Audit ran generation')
                    return g
                row={'arm':arm,'seed_offset':iteration,'generation':asdict(g),'latent_audit':json.loads(latent_path.read_text()),
                     'capture_dir':str(capture),'video_sha256':digest(g.artifacts['video']),'seconds':time.monotonic()-start}
                path=dest/'capture_receipt.json';save(path,row);key=str(path)
                clip=make_clip(row,owner.registry,owner.plans,path)
                owner.rows[key]=row;owner.clips[key]=clip
                owner.events.append({'receipt':key,'arm':arm,'seed_offset':iteration,'seconds':row['seconds']})
                save(dest.parents[0]/('completed_'+dest.name+'.json'),owner.events[-1])
                return replace(g,artifacts={**g.artifacts,'native_phase_receipt':key})
        return Capturing()
    def clip_for(self,g):
        key=g.artifacts['native_phase_receipt'];row=self.rows[key];clip=self.clips[key]
        if g.iteration!=row['seed_offset'] or g.metadata['native_phase_contract']['arm']!=row['arm']:
            raise ValueError('Wrong capture receipt')
        if g.artifacts['video']!=row['generation']['artifacts']['video']:
            raise ValueError('Video/receipt mismatch')
        reference=self.references[g.iteration];r=self.rows[reference.artifacts['native_phase_receipt']]
        if row['arm']!='original':check_pair(r,row)
        elif row['latent_audit']['initial_noise']['sha256']!=r['latent_audit']['initial_noise']['sha256']:
            raise ValueError('Original noise mismatch')
        return clip
    def reference_for(self,g):
        return self.clips[self.references[g.iteration].artifacts['native_phase_receipt']]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--check-only',action='store_true');args=p.parse_args()
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    try:
        setup_env();registry=NativePhaseRegistry(PRE,PRE_SHA);plans=plans_for(registry);spec=specification(registry)
        save(out/'protocol.json',{'check_only':args.check_only,'specification':spec,'written_before_generation':True})
        store=CaptureStore(registry,plans,spec)
        if args.check_only:
            for arm in SCHEDULE:
                backend=NativePhaseBackend(registry,store.factory,out/'loader_checks'/arm,audit_only=True)
                request=registry.request(0,1);request=replace(request,condition={'native_phase':{**request.condition['native_phase'],'arm':arm}})
                g=backend.generate(request,0)
                assert g.metadata['dd2_audit_only'] and g.metadata['native_phase_route']['runtime_inputs_match']
                save(out/f'loader_{arm}.json',asdict(g));print('ONLINE_LOADER_PREFLIGHT_PASS',arm,flush=True)
            verify(spec,registry)
            save(out/'preflight.json',{'passed':True,'generation_run':False,'specification':spec,'actual_loader_calls':3})
            print('ONLINE_PILOT_PREFLIGHT_COMPLETE',out,flush=True);return
        gate=json.loads((PREFLIGHT/'preflight.json').read_text())
        if not gate['passed'] or gate['generation_run'] or gate['specification']!=spec:
            raise ValueError('Preflight absent or specification/code changed')
        verify(spec,registry)
        # Fresh shared references. Each of the eight method runs is charged three.
        refs=NativePhaseBackend(registry,store.factory,out/'references')
        for seed in (0,1,2):
            g=refs.generate(registry.request(0,1),seed);store.references[seed]=g
            print('REFERENCE_CAPTURED',seed,flush=True)
        detector=ReleasingDetector(DETECTOR);callback=PNGPhaseMeasurement(store.clip_for,store.reference_for,detector)
        evaluator=OnlineEvaluator(callback,detector);summaries=[]
        for target_arm in ('delayed','advanced'):
            for method in METHODS:
                dest=out/target_arm/method
                refiner=PilotRefiner(registry,method,target_arm)
                request=registry.request(registry.means[target_arm],1)
                request=replace(request,condition={'native_phase':{**request.condition['native_phase'],'arm':refiner.initial_arm()}})
                backend=NativePhaseBackend(registry,store.factory,dest/'candidates')
                runner=NativePhaseRunner(backend,evaluator,refiner,DriveLoopConfig(max_iterations=3,target_score=0,stop_on_acceptance=False,output_dir=dest/'history'))
                result=runner.run(request);save(dest/'result.json',asdict(result))
                record=method_summary(result,method,target_arm);summaries.append(record)
                save(out/'partial_summary.json',summaries);save(out/'generation_ledger.json',store.events)
                print('METHOD_COMPLETE',target_arm,method,'candidates',len(record['attempts']),'best_error',record['best_measured']['absolute_error_px'] if record['best_measured'] else None,flush=True)
        verify(spec,registry)
        actual=len(store.events);expected=3+sum(r['actual_candidate_renders'] for r in summaries)
        assert actual==expected and actual<=27
        save(out/'summary.json',{'complete':True,'development_only':True,'methods':summaries,'actual_total_renders':actual,
            'actual_shared_reference_renders':3,'logical_total_across_eight_method_runs':sum(r['logical_total_renders'] for r in summaries),
            'all_methods_completed_budget':all(r['complete_three_candidate_budget'] for r in summaries),
            'all_attempts_measurable':all(r['all_attempts_measurable'] for r in summaries),
            'paper_success_rate_computed':False,'generation_ledger':store.events})
        print('NATIVE_PHASE_ONLINE_PILOT_COMPLETE',out,flush=True)
    except Exception as exc:
        save(out/'failure.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'evidence_retained':True,'no_automatic_retry':True})
        raise

if __name__=='__main__':main()
