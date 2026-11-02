"""Frozen four-arm evaluation. prepare/check never generate; run is explicit.

Recovery replays completed generations through the unchanged evaluator/runner.
Re-evaluations are separately charged. A changed feedback request fails closed.
Incomplete GPU calls require --retry-incomplete; never choose between retries.
"""
import argparse
import fcntl
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

ROOT = Path.cwd().resolve()
sys.path.insert(0, str(ROOT))
from scripts.eval_runtime_support import atomic_json, digest, sha, SlotBackend, PresetRefiner, completion
from driveloop.schema import DriveLoopRequest, DriveLoopConfig, Generation
from driveloop.runner import DriveLoopRunner
from driveloop.refiner import RuleBasedRefiner
from driveloop.backends.drivedreamer2 import DriveDreamer2Backend

PLAN = ROOT / 'outputs/revision_20261022_eval_protocol'
SOURCE = ROOT / 'outputs/revision_20261022_eval_runtime'
AUDIT = SOURCE / 'loader_v2'
OUT = ROOT / 'outputs/frozen_eval_20261023'
METHODS = ('resample', 'direct', 'fixed', 'feedback')
BANKS = (11, 12, 13)
TENSORS = ('img_cond', 'prompt_embed', 'grounding_downsampler_input', 'box_downsampler_input')
WEIGHT = Path('/data/models/dd2/pretrained_models/drivedreamer2_img_cond/pytorch_gligen_weights.bin')
MODEL_DIRS = [Path('/data/projects/DriveLoop/hf_cache/stable-video-diffusion-img2vid-xt-1-1'),
    Path('/data/projects/DriveLoop/hf_cache/transformers/models--cerspense--zeroscope_v2_576w/snapshots/6963642a64dbefa93663d1ecebb4ceda2d9ecb28')]
CODE_DIRS = ('driveloop', 'dreamer-train', 'dreamer-models', 'dreamer-datasets')
EXTRA_CODE = ('ENV.py', 'scripts/dd2_capture_python', 'scripts/dd2_capture_hooks.py',
    'scripts/eval_runtime_support.py', 'scripts/run_frozen_eval.py', 'scripts/run_frozen_eval_20261023.sh',
    'scripts/frozen_eval_annotation.html', 'scripts/frozen_eval_annotation_rubric.txt')

def read(path):
    return json.loads(Path(path).read_text())

def seal(path, payload):
    atomic_json(path, {'payload': payload, 'sha256': digest(payload)})

def unseal(path):
    e = read(path)
    if digest(e['payload']) != e['sha256']:
        raise ValueError('Envelope changed: ' + str(path))
    return e['payload']

def verify_files(files):
    for p, h in files.items():
        if not Path(p).is_file() or sha(p) != h:
            raise ValueError('Frozen file missing/changed: ' + str(p))

def environment():
    import torch
    packages = sorted((d.metadata['Name'].lower(), d.version) for d in importlib.metadata.distributions() if d.metadata['Name'])
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,driver_version', '--format=csv,noheader'], text=True).strip()
    return {'python': platform.python_version(), 'executable': sys.executable,
        'packages': packages, 'torch_cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
        'gpu': gpu, 'machine': platform.machine(),
        'runtime_environment': {k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG',
            'NVIDIA_TF32_OVERRIDE','PYTHONHASHSEED','OMP_NUM_THREADS','MKL_NUM_THREADS','IMAGEIO_FFMPEG_EXE')},
        'parent_torch_defaults': {'deterministic': torch.are_deterministic_algorithms_enabled(),
            'cudnn_benchmark': torch.backends.cudnn.benchmark, 'cudnn_deterministic': torch.backends.cudnn.deterministic}}

def code_paths():
    return sorted({str(p.absolute()) for d in CODE_DIRS for p in (ROOT/d).rglob('*.py')
        if '.git' not in p.parts and '__pycache__' not in p.parts} | {str(ROOT/p) for p in EXTRA_CODE})

def asset_paths():
    paths = {str(WEIGHT), str(ROOT/'yolov8x.pt'), '/data/projects/DriveLoop/clip_text_transform_after_pool_panoramic.pkl'}
    for d in MODEL_DIRS:
        if not d.is_dir(): raise ValueError('Missing model directory: ' + str(d))
        paths.update(str(p) for p in d.rglob('*') if p.is_file() and '.cache' not in p.parts and '.git' not in p.parts)
    return sorted(paths)

def controls():
    return {'methods': list(METHODS), 'seed_banks': list(BANKS), 'candidate_cap': 4,
        'steps': 50, 'frame_num': 8, 'use_postprocess': False, 'intent_epsilon': .05,
        'detector_inference_confidence': .25, 'detector_postfilter_confidence': .20,
        'tau': .3, 'utility_weights': {'perception': .5, 'control': .3, 'intent': .2},
        'stop_on_acceptance': False, 'support_per_scene_bank': 1,
        'total_planned_upper_bound': 612, 'candidate_upper_bound': 576,
        'arm_order': list(METHODS), 'group_order': 'source manifest order, then seed banks 11,12,13',
        'recovery': 'Verify receipts; replay unchanged evaluator for unfinished arms; charge re-evaluations. Completed arms are verified and reused. Explicit retry of interrupted slots only.',
        'static_condition_note': 'Fixed attempts 2 and 3 and every direct attempt have identical four audited input tensors. Retain all declared seed slots.',
        'noise_rule': 'Compare actual initial_noise signature for every observed same-case/bank/iteration across methods. Mismatch stops execution; missing slots are not claimed paired.',
        'claim_boundary': 'New bounded evaluation of detector/trajectory proxies. No historical table reproduction or semantic/independent human confirmation.'}

def prepare():
    if (OUT/'freeze.json').exists():
        check(); return
    pre = read(SOURCE/'source_preflight.json')
    verify_files(pre['source_hashes'])
    audit = read(AUDIT/'loader_audit_results.json')
    if not audit['complete'] or len(audit['rows']) != 60 or audit['generation_run']:
        raise ValueError('Incomplete input audits')
    # All upstream audited code/source identities must still match.
    verify_files(read(AUDIT/'loader_audit_protocol.json')['inputs'])
    paths = set(code_paths()) | set(asset_paths()) | set(pre['source_hashes'])
    paths.update(str(p) for p in PLAN.glob('*.json'))
    paths.update(str(p) for p in [SOURCE/'source_preflight.json', AUDIT/'loader_audit_protocol.json', AUDIT/'loader_audit_results.json'])
    for c in pre['cases']:
        paths.update([c['manifest'], str(AUDIT/c['case_id']/'effective_request_plan.json')])
    for r in audit['rows']:
        payload = unseal(r['receipt']); verify_files(payload['files'])
        paths.add(r['receipt']); paths.update(payload['files'])
    print('FREEZING', len(paths), 'files (including model weights)', flush=True)
    files = {p: sha(p) for p in sorted(paths)}
    groups = [{'case_id': c['case_id'], 'bank': b} for c in pre['cases'] for b in BANKS]
    f = {'version': 1, 'created_unix': time.time(), 'controls': controls(), 'files': files,
         'code_inventory': code_paths(), 'asset_inventory': asset_paths(), 'environment': environment(),
         'groups': groups, 'assigned_arms': [dict(g, method=m) for g in groups for m in METHODS],
         'generation_started_at_freeze': False}
    seal(OUT/'freeze.json', f)
    print('FREEZE_CREATED', digest(f), '36 groups; 144 assigned arms; no generation', flush=True)

def check():
    f = unseal(OUT/'freeze.json')
    if f['controls'] != controls() or f['code_inventory'] != code_paths() or f['asset_inventory'] != asset_paths():
        raise ValueError('Runtime definition or file inventory changed')
    if digest(f['environment']) != digest(environment()):
        raise ValueError('Frozen environment changed')
    verify_files(f['files'])
    print('FROZEN_INPUTS_VERIFIED', len(f['files']), 'files; no generation in this check', flush=True)
    return f

def stat_snapshot(files):
    return {p: (Path(p).stat().st_size, Path(p).stat().st_mtime_ns, Path(p).stat().st_ctime_ns) for p in files}

def validate_capture(dest):
    from scripts.dd2_capture_hooks import PIPE, TESTER
    h = read(dest/'capture/hook_state.json')
    if sorted(h['installed']) != sorted([PIPE, TESTER]): raise ValueError('Capture hooks changed')
    if h['counts'] != {'decode': 1, 'postprocess': 1, 'composite': 1}: raise ValueError('Capture call counts')
    if h['capture_rng_unchanged'] != ['before_decode','after_decode','after_postprocess','after_composite']:
        raise ValueError('Capture changed RNG')
    if not h['generated_equals_composite_bottom']: raise ValueError('Composite mismatch')
    names = {'final_latents.pt', 'decoded_before_postprocess.pt'} | {
        f'{d}/frame{i}.png' for d in ('generated_panorama', 'generated_front', 'composite_before_mp4') for i in range(8)}
    if set(h['files']) != names: raise ValueError('Capture inventory')
    verify_files({str(dest/'capture'/p): v for p,v in h['files'].items()})
    if h['decode_settings'] != {'num_frames':8,'decode_chunk_size':1,'clip_latents':False,'vae_dtype':'torch.float16','scaling_factor':.18215}:
        raise ValueError('Decode settings changed')
    a = read(dest/'latent_audit.json')
    if a['scale'] != 1.0 or a['image_latent_before'] != a['image_latent_after']:
        raise ValueError('Image conditioning changed')
    n = a['initial_noise']
    if n['shape'] != [1,8,4,32,336] or n['dtype'] != 'float16' or len(n['sha256']) != 64 or n['nonzero'] <= 0:
        raise ValueError('Invalid noise signature')
    return n

def pair_noise(group, method, iteration, noise):
    p = group/'noise_pairs'/f'{iteration:02d}.json'
    row = unseal(p) if p.exists() else {'iteration': iteration, 'observed': {}}
    if any(n != noise for n in row['observed'].values()):
        raise ValueError('Actual initial-noise mismatch across methods at ' + str(p))
    row['observed'][method] = noise
    row['all_four_observed'] = set(row['observed']) == set(METHODS)
    seal(p, row)

def generation_validator(manifest, bank, method, group, audits):
    def validate(g, dest):
        if not Path(g.artifacts.get('video','')).is_file() or g.metadata.get('dd2_audit_only'):
            raise ValueError('No actual video')
        binding = g.metadata['dd2_source_sample_binding']
        if not binding['ready']: raise ValueError('Source unbound')
        for k in ('scene_token', 'instance_token', 'source_candidate_id'):
            if binding['selector'][k] != manifest[k]: raise ValueError('Source changed: ' + k)
        params = g.metadata['dd2_generation_parameter_env']
        if params.get('DRIVELOOP_DD2_SEED_OFFSET') != str(bank*100+g.iteration) or params.get('DRIVELOOP_DD2_NUM_INF_STEPS') != '50':
            raise ValueError('Steps/seed offset changed')
        if set(params)-{'DRIVELOOP_DD2_SEED_OFFSET','DRIVELOOP_DD2_NUM_INF_STEPS'}:
            raise ValueError('Unexpected guidance override')
        req = read(dest/'started.json')['request']
        key = digest(req)
        a = g.metadata['dd2_runtime_input_audit']
        expected = audits.get(key)
        if method != 'feedback' and expected is None: raise ValueError('Static request not previously audited')
        for k in TENSORS:
            s = a[k]
            if not s.get('available') or s.get('nonzero',0)<=0 or len(s.get('sha256',''))!=64:
                raise ValueError('Unavailable/empty runtime tensor: '+k)
            if any(not math.isfinite(s[v]) for v in ('sum','mean','std')): raise ValueError('Nonfinite input summary')
            if expected and s['sha256'] != expected[k]['sha256']: raise ValueError('Static tensor drift: '+k)
            shapes={'img_cond':[1,3,256,2688],'grounding_downsampler_input':[8,3,256,2688],
                'box_downsampler_input':[8,19,256,2688],'prompt_embed':[48,1,1024] if method=='support' else [1,1,1024]}
            if s['shape']!=shapes[k] or s['dtype']!='float32': raise ValueError('Unexpected tensor shape/dtype: '+k)
        # Source pixels/maps must stay fixed even on previously unseen feedback branches.
        base = next(iter(audits.values()))
        for k in ('img_cond','grounding_downsampler_input'):
            if a[k]['sha256'] != base[k]['sha256']: raise ValueError('Source image/map drift')
        if method == 'support' and g.metadata['dd2_override_json']['available']:
            raise ValueError('Support was injected')
        n = validate_capture(dest)
        if method != 'support': pair_noise(group, method, g.iteration, n)
    return validate

class CheckedBackend:
    def __init__(self, slots, frozen_stats, trace):
        self.slots, self.frozen_stats, self.trace = slots, frozen_stats, trace
    def generate(self, request, iteration):
        if stat_snapshot(self.frozen_stats) != self.frozen_stats: raise ValueError('Frozen input changed during run')
        if self.trace is not None and iteration in self.trace and digest(asdict(request)) != self.trace[iteration]:
            raise ValueError('Effective request differs from predeclared trace')
        return self.slots.generate(request, iteration)

def instrument_evaluator(evaluator, history, prior_evaluations=None):
    original = evaluator.evaluate
    counter = [0]
    def measured(g):
        i=counter[0];counter[0]+=1
        p=history/'evaluation_calls'/f'{i:02d}.json'
        row={'iteration':g.iteration,'started_unix':time.time(),'complete':False}
        atomic_json(p,row);start=time.monotonic()
        try:
            value=original(g);row['evaluation']=asdict(value)
            if prior_evaluations and g.iteration in prior_evaluations:
                if digest(asdict(value))!=digest(prior_evaluations[g.iteration]):
                    raise ValueError('Frozen evaluator changed on generation replay')
            row['complete']=True
            return value
        except Exception as e:
            row['error']={'type':type(e).__name__,'message':str(e)};raise
        finally:
            row['wall_seconds']=time.monotonic()-start
            row['detector_runtime']=evaluator.detector.get_runtime_audit()
            atomic_json(p,row)
    evaluator.evaluate=measured

def ledger(f):
    rows=[]
    for assigned in f['assigned_arms']:
        arm=OUT/'groups'/assigned['case_id']/str(assigned['bank'])/assigned['method']
        p=arm/'outcome.json'
        state=unseal(p)['completion'] if p.exists() else {'complete':False,'termination':'incomplete_execution' if arm.exists() else 'not_started'}
        support=arm.parent/'support/slots/slot_00'
        if not p.exists() and not (support/'receipt.json').exists() and list(support.glob('try_*/started.json')):
            state={'complete':False,'termination':'blocked_by_incomplete_support'}
        rows.append(dict(assigned, **state))
    starts=list((OUT/'groups').glob('*/*/*/slots/slot_*/try_*/started.json'))
    receipts=list((OUT/'groups').glob('*/*/*/slots/slot_*/receipt.json'))
    evals=[read(p) for p in (OUT/'groups').glob('*/*/*/history_*/evaluation_calls/*.json')]
    tries={str(p.parent.parent) for p in starts}
    noise=[unseal(p) for p in (OUT/'groups').glob('*/*/noise_pairs/*.json')]
    payload={'assigned_arms':len(f['assigned_arms']),'complete_arms':sum(r['complete'] for r in rows),'rows':rows,
        'generation_calls_started':len(starts),'completed_generation_receipts':len(receipts),
        'support_receipts':sum('/support/' in str(p) for p in receipts),
        'extra_retry_calls_started':len(starts)-len(tries),
        'generation_completed_wall_seconds':sum(unseal(p)['wall_seconds'] for p in receipts),
        'evaluation_calls_started':len(evals),'evaluation_calls_completed':sum(r['complete'] for r in evals),
        'successful_detector_predict_calls':sum(r.get('detector_runtime',{}).get('successful_predict_calls',0) for r in evals),
        'evaluation_wall_seconds':sum(r.get('wall_seconds',0) for r in evals),
        'fully_observed_noise_slots':sum(n['all_four_observed'] for n in noise),
        'partly_observed_noise_slots':sum(not n['all_four_observed'] for n in noise),
        'cost_caveat':'Started calls include failures; hard-killed GPU/detector calls may lack final timing/counters. Completed receipt time includes validation. Evaluation replay calls remain charged.',
        'analysis_status':'No efficacy analysis here; not_started/incomplete remain unknown, never silently zero.'}
    atomic_json(OUT/'ledger.json',payload)
    return payload

def run(f, groups, retry):
    from driveloop.perception_video import UltralyticsYOLODetector
    from driveloop.perception_v10 import ManeuverViewRestrictedSuperclassEvaluator
    pre=read(SOURCE/'source_preflight.json');recipes=read(PLAN/'declared_method_requests.json')
    frozen_stats=stat_snapshot(f['files']);identity=digest(f)
    for k in list(os.environ):
        if k.startswith('DRIVELOOP_'): del os.environ[k]
    os.environ.update(DRIVELOOP_DD2_NUM_INF_STEPS='50',DRIVELOOP_EGO_INJECTION='1',
        DRIVELOOP_DD2_WEIGHT_PATH=str(WEIGHT),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    audit_rows=read(AUDIT/'loader_audit_results.json')['rows']
    for assigned in f['groups'][:groups]:
        case,bank=assigned['case_id'],assigned['bank'];group=OUT/'groups'/case/str(bank)
        manifest=read(next(c['manifest'] for c in pre['cases'] if c['case_id']==case))
        recipe=next(r for r in recipes['rows'] if r['case_id']==case)
        audits={r['request_sha256']:r['audit'] for r in audit_rows if r['case_id']==case}
        traces=read(AUDIT/case/'effective_request_plan.json')['requests']
        os.environ['DRIVELOOP_DD2_SEED_BANK']=str(bank)
        baseline=None
        for method in ('support',)+METHODS:
            arm=group/method
            validate=generation_validator(manifest,bank,method,group,audits)
            def factory(dest):
                os.environ['DRIVELOOP_CAPTURE_DIR']=str(dest/'capture')
                os.environ['DRIVELOOP_DD2_IMAGE_CONDITION_AUDIT_PATH']=str(dest/'latent_audit.json')
                return DriveDreamer2Backend(project_root=ROOT,baseline_dataset_dir=manifest['dataset'],artifact_dir=dest/'artifacts',
                    python_executable=str(ROOT/'scripts/dd2_capture_python'),timeout_seconds=3600,
                    source_candidate_id=manifest['source_candidate_id'],scene_token=manifest['scene_token'],instance_token=manifest['instance_token'],
                    source_identity_summary_path=manifest['identity'])
            slots=SlotBackend(arm/'slots',digest({'freeze':identity,'case':case,'bank':bank,'method':method}),factory,retry,validate)
            if method=='support':
                baseline=CheckedBackend(slots,frozen_stats,None).generate(DriveLoopRequest(
                    prompt='Uninjected source-scene support baseline',condition={'generation_escalation':{'num_inf_steps':50}}),0)
                ledger(f);continue
            outcome=arm/'outcome.json'
            if outcome.exists():
                saved=unseal(outcome)
                if saved['freeze_sha256']!=identity: raise ValueError('Outcome from another freeze')
                verify_files(saved['files'])
                for p in sorted((arm/'slots').glob('slot_*/receipt.json')):
                    data=unseal(p);verify_files(data['files']);validate(Generation(**data['generation']),Path(data['try_directory']))
                print('REUSED_ARM',case,bank,method,flush=True);continue
            reqs=[DriveLoopRequest(**r) for r in recipe['methods'][method]]
            trace={u['iteration']:key for key,r in traces.items() for u in r['used_by'] if u['method']=='resample' and u['iteration']==0}
            if method!='feedback':
                trace={u['iteration']:key for key,r in traces.items() for u in r['used_by'] if u['method']==method}
                if set(trace)!=set(range(4)): raise ValueError('Incomplete static request trace')
            history=arm/f'history_{time.time_ns()}'
            evaluator=ManeuverViewRestrictedSuperclassEvaluator(
                detector=UltralyticsYOLODetector(str(ROOT/'yolov8x.pt'),confidence_threshold=.20,inference_confidence_threshold=.25),
                confidence_threshold=.20,baseline_video=baseline.artifacts['video'])
            prior={}
            for p in sorted(arm.glob('history_*/evaluation_calls/*.json')):
                r=read(p)
                if r['complete']:
                    if r['iteration'] in prior and digest(prior[r['iteration']])!=digest(r['evaluation']):
                        raise ValueError('Conflicting completed evaluations in recovery history')
                    prior[r['iteration']]=r['evaluation']
            instrument_evaluator(evaluator,history,prior)
            runner=DriveLoopRunner(CheckedBackend(slots,frozen_stats,trace),evaluator,
                RuleBasedRefiner() if method=='feedback' else PresetRefiner(reqs),
                config=DriveLoopConfig(max_iterations=4,target_score=.99,output_dir=history,use_task_utility=True,
                    utility_weights={'perception':.5,'control':.3,'intent':.2},use_postprocess=False,intent_epsilon=.05,
                    acceptance_policy='joint_perception_direction',perception_acceptance_threshold=.3,stop_on_acceptance=False))
            try:
                result=runner.run(reqs[0]);done=completion(result)
            finally:
                evaluator.detector.release_gpu()
            files={str(p):sha(p) for p in history.rglob('*') if p.is_file()}
            files.update({str(p):sha(p) for p in (arm/'slots').glob('slot_*/receipt.json')})
            seal(outcome,{'freeze_sha256':identity,'completion':done,'result':asdict(result),'files':files,'generation_events':slots.events})
            ledger(f)
            print('ARM_COMPLETE',case,bank,method,done,flush=True)
        print('GROUP_COMPLETE',case,bank,flush=True)
    if stat_snapshot(f['files'])!=frozen_stats: raise ValueError('Frozen inputs changed during execution')
    value=ledger(f)
    print('ALL_ASSIGNED_ARMS_COMPLETE' if value['complete_arms']==144 else 'BOUNDED_PREFIX_COMPLETE',value['complete_arms'],'of 144 arms',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','check','run'))
    p.add_argument('--groups',type=int,default=1,help='Fixed prefix of 36 planned scene-bank groups, default 1 (up to 17 renders).')
    p.add_argument('--retry-incomplete',action='store_true');args=p.parse_args()
    if not 1<=args.groups<=36: p.error('groups must be 1..36')
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'outputs/dd2_formal_generation.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # All entrypoint invocations are recorded; this lock coordinates this harness.
    # Older standalone scripts do not honor it and must not be run concurrently.
    invocation=OUT/'invocations'/f'{time.time_ns()}.json'
    record={'stage':args.stage,'groups':args.groups,'retry_incomplete':args.retry_incomplete,'start_unix':time.time(),'complete':False}
    atomic_json(invocation,record)
    f=None
    try:
        if args.stage=='prepare': prepare()
        else:
            f=check()
            if args.stage=='run': run(f,args.groups,args.retry_incomplete)
        record['complete']=True
    except Exception as e:
        record['error']={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
        raise
    finally:
        record['end_unix']=time.time();record['wall_seconds']=record['end_unix']-record['start_unix']
        atomic_json(invocation,record)
        if f is not None and args.stage=='run': ledger(f)

if __name__=='__main__': main()
