"""Post-hoc adaptive structural ablations. prepare is CPU-only; run is explicit.

Frozen Full/support results are read-only references. No synthetic intervention
includes static drafts and automatic fallback, starting at slot zero.
"""
import argparse
import copy
import fcntl
import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

ROOT=Path('/home/lesliton/projects/DriveLoop')
OUT=Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(OUT))
from scripts import run_frozen_eval as ref
from scripts.eval_runtime_support import atomic_json,digest,sha,SlotBackend,completion
from driveloop.schema import DriveLoopRequest,DriveLoopConfig,Generation
from driveloop.runner import DriveLoopRunner
from driveloop.refiner import RuleBasedRefiner
from driveloop.backends.drivedreamer2 import DriveDreamer2Backend
from isolated_backend import NoSyntheticBackend,BACKEND_VERSION
from isolated_refiner import IsolatedRefiner

PRE=ROOT/'outputs/revision_20261027_module_preflight_v2'
EXPECTED='5a4b765b63d6a606d1e3ab197060e448a39f422423abe282657a63280076989a'
METHODS=('without_synthetic','without_size')
read,unseal,seal=ref.read,ref.unseal,ref.seal

def prepare():
    f=unseal(ref.OUT/'freeze.json')
    assert digest(f)==EXPECTED
    # Full expensive model/artifact verification is repeated by explicit run.
    ref.verify_files({p:f['files'][p] for p in f['code_inventory']})
    v=read(PRE/'validation.json')
    assert v['complete'] and v['checked_unique_requests']==210 and v['reference_requests_matched']==144
    ref.verify_files({str(PRE/name):h for name,h in v['files'].items()})
    plan=read(PRE/'request_plan.json');observed=read(PRE/'observed_inputs.json')
    assert set(plan['unique_requests'])==set(observed)
    assert sha(OUT/'isolated_backend.py')==v['files']['isolated_backend.py']
    assert sha(OUT/'isolated_refiner.py')==v['files']['isolated_refiner.py']
    paths={OUT/n for n in ('run_module_ablation.py','isolated_backend.py','isolated_refiner.py','run_pilot.sh')}
    paths.update(PRE/n for n in ('validation.json','request_plan.json','observed_inputs.json','protocol.json'))
    paths.add(ref.OUT/'freeze.json')
    original_requests=0
    for g in f['groups']:
        group=ref.OUT/'groups'/g['case_id']/str(g['bank'])
        op=group/'feedback/outcome.json';outcome=unseal(op)
        assert outcome['freeze_sha256']==EXPECTED and outcome['completion']['complete']
        assert outcome['completion']['attempt_count']==4
        paths.add(op)
        for method,n in (('feedback',4),('support',1)):
            for slot in range(n):
                rp=group/method/'slots'/f'slot_{slot:02d}'/'receipt.json'
                receipt=unseal(rp);paths.add(rp)
                if method!='feedback':continue
                start=Path(receipt['try_directory'])/'started.json'
                assert sha(start)==receipt['files'][str(start)]
                row=next(r for r in plan['comparisons'] if r['case_id']==g['case_id'] and
                         r['bank']==g['bank'] and r['next_iteration']==slot)
                expected_request=plan['unique_requests'][g['case_id']+':'+row['keys']['full']]['request']
                assert digest(read(start)['request'])==digest(expected_request)
                original_requests+=1
    protocol={'version':1,'reference_freeze':EXPECTED,'groups':f['groups'],
        'methods':list(METHODS),'candidate_cap':4,'steps':50,
        'files':{str(p):sha(p) for p in sorted(paths)},
        'full_effective_requests_verified':original_requests,
        'controls':ref.controls(),
        'new_arms':72,'new_candidate_upper_bound':288,
        'reuse_full_candidates':144,'reuse_support':36,
        'scope':'Post-hoc adaptive module ablation on previously inspected scenes; not fresh held-out validation.',
        'synthetic_intervention':'Remove synthetic static/trajectory injections from slot0; retain verified real-track reinforcement.',
        'size_intervention':'Set refiner structural size_scale to1, retain reference prompt/reset and other actions.',
        'seed_rule':'bank*100+iteration; compare captured initial noise to corresponding frozen Full slot.',
        'same_initial_output_assumed':False,
        'unseen_requests':'Validate source/shape/provenance/noise at runtime; compare audited tensors when request is in preflight.',
        'recovery':'Completed slots verified/replayed. Interrupted slots require explicit --retry-incomplete. Re-evaluations charged.',
        'semantic_status':'Detector proxies only until separate blinded human review; no superiority assumption.'}
    target=OUT/'protocol.json'
    if target.exists():assert unseal(target)==protocol,'Protocol changed; use a new version'
    else:seal(target,protocol)
    print('MODULE_PROTOCOL_READY',digest(protocol),'144 Full requests matched; no generation',flush=True)
    return protocol

def check(p):
    f=ref.check()
    assert digest(f)==EXPECTED
    ref.verify_files(p['files'])
    return f

def ledger(p):
    rows=[]
    for g in p['groups']:
        for m in METHODS:
            path=OUT/'groups'/g['case_id']/str(g['bank'])/m/'outcome.json'
            c=unseal(path)['completion'] if path.exists() else {'complete':False,'termination':'not_completed'}
            rows.append(dict(g,method=m,**c))
    starts=list((OUT/'groups').glob('*/*/*/slots/slot_*/try_*/started.json'))
    receipts=list((OUT/'groups').glob('*/*/*/slots/slot_*/receipt.json'))
    evals=[read(pth) for pth in (OUT/'groups').glob('*/*/*/history_*/evaluation_calls/*.json')]
    data={'assigned_new_arms':72,'complete_new_arms':sum(r['complete'] for r in rows),'rows':rows,
        'new_generation_calls_started':len(starts),'new_completed_receipts':len(receipts),
        'extra_retry_starts':len(starts)-len({str(q.parent.parent) for q in starts}),
        'evaluation_calls_started':len(evals),'evaluation_calls_completed':sum(e['complete'] for e in evals),
        'evaluation_wall_seconds':sum(e.get('wall_seconds',0) for e in evals),
        'generation_completed_wall_seconds':sum(unseal(q)['wall_seconds'] for q in receipts),
        'reference_candidates_available':144,'reference_support_available':36,
        'cost_note':'Reused reference compute must still be counted in comparative method budgets; new-start counts are additional cost only.',
        'semantic_efficacy_evaluated':False}
    atomic_json(OUT/'ledger.json',data)
    return data

def validator(manifest,bank,mode,group,known,source,noise):
    def validate(g,dest):
        assert Path(g.artifacts.get('video','')).is_file() and not g.metadata.get('dd2_audit_only')
        binding=g.metadata['dd2_source_sample_binding'];assert binding['ready']
        for k in ('scene_token','instance_token','source_candidate_id'):
            assert binding['selector'][k]==manifest[k]
        assert g.metadata['dd2_generation_parameter_env']=={
            'DRIVELOOP_DD2_SEED_OFFSET':str(bank*100+g.iteration),'DRIVELOOP_DD2_NUM_INF_STEPS':'50'}
        req=read(dest/'started.json')['request'];a=g.metadata['dd2_runtime_input_audit']
        key=digest(req);expected=known.get(key)
        shapes={'img_cond':[1,3,256,2688],'grounding_downsampler_input':[8,3,256,2688],
                'box_downsampler_input':[8,19,256,2688],'prompt_embed':[1,1,1024]}
        for k in ref.TENSORS:
            t=a[k]
            assert t['available'] and t['nonzero']>0 and len(t['sha256'])==64
            assert t['shape']==shapes[k] and t['dtype']=='float32'
            assert all(math.isfinite(t[v]) for v in ('sum','mean','std'))
            if expected:assert t['sha256']==expected[k]['sha256'],'Audited tensor drift: '+k
        for k in ('img_cond','grounding_downsampler_input'):
            assert a[k]['sha256']==source[k]['sha256'],'Source image/map drift'
        override=g.metadata['dd2_override_json']
        if mode=='without_synthetic':
            boxes=override['boxes3d'];audit=override['audit']['synthetic_intervention']
            assert audit['version']==BACKEND_VERSION
            assert boxes['append']==boxes['per_frame_append']==[]
            assert all(e['source']=='source_bound_real_track' and e['synthetic_track_id'] is None for e in boxes['per_frame_append_ego'])
        else:
            scale=req['condition'].get('structural_escalation',{}).get('size_scale',1.0)
            assert scale==1.0,'Size escalation remains enabled'
        captured=ref.validate_capture(dest)
        assert captured==noise[g.iteration],'Initial noise differs from frozen Full'
        proof=group/'noise_pairs'/mode/f'{g.iteration:02d}.json'
        value={'iteration':g.iteration,'mode':mode,'noise':captured,'matched_frozen_full':True}
        if proof.exists():assert unseal(proof)==value
        else:seal(proof,value)
    return validate

def run(p,f,groups,retry):
    from driveloop.perception_video import UltralyticsYOLODetector
    from driveloop.perception_v10 import ManeuverViewRestrictedSuperclassEvaluator
    pre=read(ref.SOURCE/'source_preflight.json');recipes=read(ref.PLAN/'declared_method_requests.json')
    plan=read(PRE/'request_plan.json');observed=read(PRE/'observed_inputs.json')
    all_files=dict(f['files']);all_files.update(p['files']);stats=ref.stat_snapshot(all_files)
    for k in list(os.environ):
        if k.startswith('DRIVELOOP_'):del os.environ[k]
    os.environ.update(DRIVELOOP_DD2_NUM_INF_STEPS='50',DRIVELOOP_EGO_INJECTION='1',
        DRIVELOOP_DD2_WEIGHT_PATH=str(ref.WEIGHT),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    for assigned in p['groups'][:groups]:
        case,bank=assigned['case_id'],assigned['bank'];group=OUT/'groups'/case/str(bank)
        old=ref.OUT/'groups'/case/str(bank)
        manifest=read(next(c['manifest'] for c in pre['cases'] if c['case_id']==case))
        recipe=next(r for r in recipes['rows'] if r['case_id']==case)
        os.environ['DRIVELOOP_DD2_SEED_BANK']=str(bank)
        noise={};reused=[]
        for m,n in (('support',1),('feedback',4)):
            for i in range(n):
                path=old/m/'slots'/f'slot_{i:02d}'/'receipt.json';receipt=unseal(path)
                ref.verify_files(receipt['files'])
                g=Generation(**receipt['generation']);actual=ref.validate_capture(Path(receipt['try_directory']))
                if m=='support':baseline=g
                else:noise[i]=actual
                reused.append({'method':m,'iteration':i,'receipt':str(path),'sha256':sha(path)})
        atomic_json(group/'reference_reuse.json',reused)
        for mode in METHODS:
            backend_mode=BACKEND_VERSION if mode=='without_synthetic' else 'frozen_reference'
            known={r['request_sha256']:observed[k]['input_audit'] for k,r in plan['unique_requests'].items()
                   if r['case_id']==case and r['backend_mode']==backend_mode}
            source=next(iter(known.values()))
            arm=group/mode;outcome=arm/'outcome.json'
            validate=validator(manifest,bank,mode,group,known,source,noise)
            if outcome.exists():
                saved=unseal(outcome);assert saved['protocol_sha256']==digest(p)
                ref.verify_files(saved['files'])
                for path in sorted((arm/'slots').glob('slot_*/receipt.json')):
                    receipt=unseal(path);ref.verify_files(receipt['files'])
                    validate(Generation(**receipt['generation']),Path(receipt['try_directory']))
                print('REUSED_MODULE_ARM',case,bank,mode,flush=True);continue
            def factory(dest):
                os.environ['DRIVELOOP_CAPTURE_DIR']=str(dest/'capture')
                os.environ['DRIVELOOP_DD2_IMAGE_CONDITION_AUDIT_PATH']=str(dest/'latent_audit.json')
                cls=NoSyntheticBackend if mode=='without_synthetic' else DriveDreamer2Backend
                return cls(project_root=ROOT,baseline_dataset_dir=manifest['dataset'],artifact_dir=dest/'artifacts',
                    python_executable=str(ROOT/'scripts/dd2_capture_python'),timeout_seconds=3600,
                    source_candidate_id=manifest['source_candidate_id'],scene_token=manifest['scene_token'],
                    instance_token=manifest['instance_token'],source_identity_summary_path=manifest['identity'])
            slots=SlotBackend(arm/'slots',digest({'protocol':digest(p),'case':case,'bank':bank,'method':mode}),factory,retry,validate)
            request=DriveLoopRequest(**copy.deepcopy(recipe['methods']['feedback'][0]))
            if mode=='without_synthetic':request.condition.pop('synthetic_trajectory_escalation',None)
            elif isinstance(request.condition.get('structural_escalation'),dict):request.condition['structural_escalation']['size_scale']=1.0
            history=arm/f'history_{time.time_ns()}'
            evaluator=ManeuverViewRestrictedSuperclassEvaluator(
                detector=UltralyticsYOLODetector(str(ROOT/'yolov8x.pt'),confidence_threshold=.20,inference_confidence_threshold=.25),
                confidence_threshold=.20,baseline_video=baseline.artifacts['video'])
            prior={}
            for path in sorted(arm.glob('history_*/evaluation_calls/*.json')):
                r=read(path)
                if r['complete']:
                    if r['iteration'] in prior:assert digest(prior[r['iteration']])==digest(r['evaluation'])
                    prior[r['iteration']]=r['evaluation']
            ref.instrument_evaluator(evaluator,history,prior)
            runner=DriveLoopRunner(ref.CheckedBackend(slots,stats,None),evaluator,
                IsolatedRefiner(RuleBasedRefiner(),mode),
                config=DriveLoopConfig(max_iterations=4,target_score=.99,output_dir=history,use_task_utility=True,
                    utility_weights={'perception':.5,'control':.3,'intent':.2},use_postprocess=False,intent_epsilon=.05,
                    acceptance_policy='joint_perception_direction',perception_acceptance_threshold=.3,stop_on_acceptance=False))
            try:result=runner.run(request);done=completion(result)
            finally:evaluator.detector.release_gpu()
            files={str(path):sha(path) for path in history.rglob('*') if path.is_file()}
            files.update({str(path):sha(path) for path in (arm/'slots').glob('slot_*/receipt.json')})
            seal(outcome,{'protocol_sha256':digest(p),'completion':done,'result':asdict(result),'files':files,'generation_events':slots.events})
            ledger(p);print('MODULE_ARM_COMPLETE',case,bank,mode,done,flush=True)
        print('MODULE_GROUP_COMPLETE',case,bank,flush=True)
    assert ref.stat_snapshot(all_files)==stats
    l=ledger(p)
    print('ALL_MODULE_ARMS_COMPLETE' if l['complete_new_arms']==72 else 'MODULE_BOUNDED_PREFIX_COMPLETE',
          l['complete_new_arms'],'of 72 new arms',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','run'))
    parser.add_argument('--groups',type=int,default=1);parser.add_argument('--retry-incomplete',action='store_true')
    a=parser.parse_args();assert 1<=a.groups<=36
    lock=(ROOT/'outputs/dd2_formal_generation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    invocation=OUT/'invocations'/f'{time.time_ns()}.json'
    rec={'stage':a.stage,'groups':a.groups,'retry_incomplete':a.retry_incomplete,'complete':False,'start_unix':time.time()}
    atomic_json(invocation,rec);p=None
    try:
        p=prepare()
        if a.stage=='run':run(p,check(p),a.groups,a.retry_incomplete)
        rec['complete']=True
    except Exception as e:
        rec['error']={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()};raise
    finally:
        rec['end_unix']=time.time();atomic_json(invocation,rec)
        if p and a.stage=='run':ledger(p)

if __name__=='__main__':main()
