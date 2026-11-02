"""Actual DD2 loader/transform audit; CUDA and CLIP required, no DD2 video generation.

Default: all selected sources. --max-new N bounds newly executed loader calls.
Completed receipts are verified and reused; failures retain logs and stop.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import fcntl
from dataclasses import asdict
from pathlib import Path

ROOT=Path.cwd().resolve();sys.path.insert(0,str(ROOT))
from scripts.eval_runtime_support import atomic_json, digest, sha, PresetRefiner, SlotBackend
from driveloop.schema import DriveLoopRequest, DriveLoopConfig, Evaluation, Diagnosis, Generation
from driveloop.runner import DriveLoopRunner
from driveloop.backends.drivedreamer2 import DriveDreamer2Backend

PLAN=ROOT/'outputs/revision_20261022_eval_protocol'
OUT=ROOT/'outputs/revision_20261022_eval_runtime'
AUDIT=OUT/'loader_v2'

class TraceBackend:
    def __init__(self): self.requests=[]
    def generate(self,r,i):
        self.requests.append(asdict(r))
        return Generation(i,r.prompt,{}, {'synthetic_preflight_only':True})

class TraceEvaluator:
    def evaluate(self,g):
        # Declared stimulus to exercise deterministic fixed schedules only.
        # It is NEVER saved or reported as an experimental measurement.
        return Evaluation(0.0,{'maneuver_direction_consistent':False},Diagnosis(False,['target_object_not_detected','low_detection_coverage']))

def trace_requests(case,recipes):
    rows=[]
    for method in ('resample','direct','fixed'):
        reqs=[DriveLoopRequest(**r) for r in recipes['methods'][method]]
        backend=TraceBackend()
        history=AUDIT/case/'synthetic_request_traces'/method
        if history.exists():
            # Logger files are synthetic audit trace, never experimental data.
            history=history/str(time.time_ns())
        runner=DriveLoopRunner(backend,TraceEvaluator(),PresetRefiner(reqs),
            config=DriveLoopConfig(max_iterations=4,use_task_utility=True,acceptance_policy='joint_perception_direction',
                perception_acceptance_threshold=.3,stop_on_acceptance=False,output_dir=history))
        runner.run(reqs[0])
        rows.extend({'method':method,'iteration':i,'request':r} for i,r in enumerate(backend.requests))
    assert rows[0]['request']==next(r['request'] for r in rows if r['method']=='fixed' and r['iteration']==0)
    unique={}
    for row in rows:
        key=digest(row['request']);unique.setdefault(key,{'request':row['request'],'used_by':[]})['used_by'].append({k:row[k] for k in ('method','iteration')})
    baseline=asdict(DriveLoopRequest(prompt='Uninjected source-scene support baseline',condition={'generation_escalation':{'num_inf_steps':50}}))
    unique[digest(baseline)]={'request':baseline,'used_by':[{'method':'support','iteration':0}]}
    return unique

def validate(g,dest):
    if g.artifacts.get('video') or g.metadata.get('dd2_audit_only') is not True:raise ValueError('Unexpected generation; audit must not render')
    binding=g.metadata['dd2_source_sample_binding']
    if binding.get('ready') is not True:raise ValueError('Unbound source')
    a=g.metadata['dd2_runtime_input_audit']
    for key in ('img_cond','prompt_embed','grounding_downsampler_input','box_downsampler_input'):
        if not a[key].get('available') or not a[key].get('sha256') or a[key].get('nonzero',0)<=0:raise ValueError('Invalid runtime tensor: '+key)
    h=json.loads((dest/'capture/hook_state.json').read_text())
    if h['counts']!={'decode':0,'postprocess':0,'composite':0}:raise ValueError('Unexpected generation/decode')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--max-new',type=int);parser.add_argument('--retry-incomplete',action='store_true');args=parser.parse_args()
    for k in list(os.environ):
        if k.startswith('DRIVELOOP_'):del os.environ[k]
    os.environ.update(DRIVELOOP_DD2_NUM_INF_STEPS='50',DRIVELOOP_DD2_SEED_BANK='11',
        DRIVELOOP_EGO_INJECTION='1',DRIVELOOP_DD2_AUDIT_ONLY='1',
        DRIVELOOP_DD2_WEIGHT_PATH='/data/models/dd2/pretrained_models/drivedreamer2_img_cond/pytorch_gligen_weights.bin',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    AUDIT.mkdir(exist_ok=True)
    pre=json.loads((OUT/'source_preflight.json').read_text())
    for p,h in pre['source_hashes'].items():
        if sha(p)!=h:raise ValueError('Source changed: '+p)
    recipes=json.loads((PLAN/'declared_method_requests.json').read_text())
    assert recipes['selected_sources_sha256']==pre['selection_sha256']
    snapshots={}
    for folder in ['driveloop','dreamer-train/projects/DriveDreamer2','dreamer-models/dreamer_models','dreamer-datasets/dreamer_datasets']:
        snapshots.update({str(p):sha(p) for p in (ROOT/folder).rglob('*.py')})
    for p in [ROOT/'scripts/eval_runtime_support.py',Path(__file__),ROOT/'scripts/dd2_capture_hooks.py',ROOT/'scripts/dd2_capture_python',ROOT/'ENV.py',PLAN/'declared_method_requests.json',OUT/'source_preflight.json']:
        snapshots[str(p)]=sha(p)
    config={'inputs':snapshots,'generation_run':False,'audit_only':True,'cuda_and_text_encoder_required':True,'dd2_pipeline_loaded':False,
            'limits':'Checks actual transformed inputs for declared static requests. Adaptive feedback requests and actual generation noise remain checked at future runtime, not proven here.'}
    protocol=AUDIT/'loader_audit_protocol.json'
    if protocol.exists():
        if json.loads(protocol.read_text())!=config:raise ValueError('Audit code/input changed since prior run; create reviewed new audit version')
    else:atomic_json(protocol,config)
    identity=digest(config);new=0;rows=[];complete=True
    lock=(AUDIT/'loader_audit.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for c in pre['cases']:
        case=c['case_id'];manifest=json.loads(Path(c['manifest']).read_text())
        recipe=next(r for r in recipes['rows'] if r['case_id']==case)
        trace_path=AUDIT/case/'effective_request_plan.json'
        if trace_path.exists():
            saved=json.loads(trace_path.read_text());requests=saved['requests']
            if saved['audit_identity']!=identity or saved['sha256']!=digest(requests):raise ValueError('Request trace integrity failure')
        else:
            requests=trace_requests(case,recipe);atomic_json(trace_path,{'requests':requests,'audit_identity':identity,'sha256':digest(requests)})
        for key,row in requests.items():
            path=AUDIT/case/'loader_audits'/key
            exists=(path/'slot_00/receipt.json').exists()
            if not exists and args.max_new is not None and new>=args.max_new:
                complete=False;continue
            def factory(dest):
                os.environ['DRIVELOOP_CAPTURE_DIR']=str(dest/'capture')
                return DriveDreamer2Backend(project_root=ROOT,baseline_dataset_dir=manifest['dataset'],artifact_dir=dest/'artifacts',
                    python_executable=str(ROOT/'scripts/dd2_capture_python'),timeout_seconds=180,audit_only=True,
                    source_candidate_id=manifest['source_candidate_id'],scene_token=manifest['scene_token'],instance_token=manifest['instance_token'],
                    source_identity_summary_path=manifest['identity'])
            before=time.monotonic()
            backend=SlotBackend(path,digest({'audit':identity,'trace':digest(requests),'case':case}),factory,args.retry_incomplete,validate)
            g=backend.generate(DriveLoopRequest(**row['request']),0)
            assert g.metadata['dd2_source_sample_binding']['selector']['scene_token']==manifest['scene_token']
            rows.append({'case_id':case,'request_sha256':key,'used_by':row['used_by'],'receipt':str(path/'slot_00/receipt.json'),
                         'audit':g.metadata['dd2_runtime_input_audit'],'new_execution':not exists,'seconds':time.monotonic()-before})
            new+=int(not exists)
            atomic_json(AUDIT/'loader_audit_partial.json',{'generation_run':False,'complete':False,'rows':rows})
            print('LOADER_AUDIT_PASS',case,key[:10],'new',not exists,'seconds',round(rows[-1]['seconds'],2),flush=True)
    atomic_json(AUDIT/('loader_audit_results.json' if complete else 'loader_audit_bounded_results.json'),
        {'complete':complete,'generation_run':False,'cuda_and_text_encoder_required':True,'dd2_pipeline_loaded':False,'new_loader_calls':new,'checked_requests':len(rows),'rows':rows,
         'remaining':'Actual generation, noise pairing and adaptive feedback inputs are not exercised by this audit.'})
    print('ALL_LOADER_AUDITS_COMPLETE' if complete else 'BOUNDED_LOADER_AUDIT_COMPLETE',len(rows),'checked;',new,'new calls',flush=True)

if __name__=='__main__':main()
