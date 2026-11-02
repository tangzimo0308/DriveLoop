"""Synthetic-feedback command dry-run of unchanged real runner/refiner/backend. No video inference."""
import os,json,hashlib,subprocess
from pathlib import Path
from dataclasses import asdict
from unittest.mock import patch
from driveloop.runner import DriveLoopRunner
from driveloop.schema import DriveLoopRequest,DriveLoopConfig,Evaluation,Diagnosis
from driveloop.refiner import RuleBasedRefiner
from scripts.run_native_bus_control import setup_env,make_backend
ROOT=Path('/home/lesliton/projects/DriveLoop');OUT=ROOT/'outputs/revision_20261009_closed_loop_native_audit';OUT.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
files=[ROOT/'driveloop'/p for p in ('runner.py','refiner.py','condition_adapter.py','schema.py','backends/drivedreamer2.py','source_selector.py')]
files.extend([ROOT/'scripts/run_candidate1583_capture.py',ROOT/'scripts/run_native_bus_control.py'])
before={str(p):sha(p) for p in files}
pre=json.loads((ROOT/'outputs/revision_20261008_candidate1583_inputs/preflight.json').read_text());case=pre['cases'][0];manifest=case['inputs']['original'];reference={a:case['inputs'][a]['dataset'] for a in ('original','delayed','advanced')}
class FailureProbe:
 def __init__(self,reason,error):self.reason=reason;self.error=error
 def evaluate(self,g):
  return Evaluation(score=.1,metrics={'signed_lateral_error_px':self.error,'synthetic_feedback_probe':True},diagnosis=Diagnosis(passed=False,reasons=[self.reason]))
results=[]
for name,reason,error in [('static','target_appears_static',10),('alignment_positive',RuleBasedRefiner.ALIGNMENT_REASON_PREFIX+'lateral_displacement',10),('alignment_negative',RuleBasedRefiner.ALIGNMENT_REASON_PREFIX+'lateral_displacement',-10)]:
 setup_env();dest=OUT/name;calls=[]
 backend=make_backend(manifest,dest/'backend',audit_only=True)
 def record_command(cmd,**kwargs):
  e=kwargs['env'];calls.append({'command':cmd,'dataset':e.get('DRIVELOOP_DD2_DATA_OR_CONFIG'),'audit_only':e.get('DRIVELOOP_DD2_AUDIT_ONLY'),'prompt':e.get('DRIVELOOP_DD2_PROMPT'),'override_json':json.loads(e['DRIVELOOP_DD2_OVERRIDE_JSON']) if e.get('DRIVELOOP_DD2_OVERRIDE_JSON') else None,'num_inf_steps':e.get('DRIVELOOP_DD2_NUM_INF_STEPS'),'seed_offset':e.get('DRIVELOOP_DD2_SEED_OFFSET'),'selected_phase_arm':next((a for a,p in reference.items() if p==e.get('DRIVELOOP_DD2_DATA_OR_CONFIG')),None)})
  return subprocess.CompletedProcess(cmd,0)
 request=DriveLoopRequest(prompt='A bus moves through a daytime urban intersection.')
 config=DriveLoopConfig(max_iterations=3,target_score=.8,output_dir=dest/'history',stop_on_acceptance=False,use_task_utility=False)
 # Only external process invocation is intercepted: real runner, refiner, condition adapter and backend command construction execute.
 with patch('driveloop.backends.drivedreamer2.subprocess.run',side_effect=record_command):
  result=DriveLoopRunner(backend=backend,evaluator=FailureProbe(reason,error),config=config).run(request)
 attempts=[{'iteration':a.iteration,'request':asdict(a.request),'refinement':asdict(a.refinement) if a.refinement else None,'dd2_condition':a.dd2_condition,'status':a.status} for a in result.attempt_history]
 assert len(calls)==len(attempts)==3 and all(c['audit_only']=='1' for c in calls)
 assert all(c['dataset']==manifest['dataset'] for c in calls),'Unexpected native dataset routing; investigate before drawing conclusions'
 assert not list(dest.rglob('*.mp4'))
 results.append({'name':name,'synthetic_reason':reason,'synthetic_error_px':error,'attempts':attempts,'commands':calls,'return_status':result.return_status})
 print('DRY_RUN',name,'datasets',[c['selected_phase_arm'] for c in calls],'condition_keys',[sorted(a['refinement']['condition']) for a in attempts],flush=True)
pos,neg=results[1:]
same=all(a['refinement']==b['refinement'] and a['dd2_condition']==b['dd2_condition'] for a,b in zip(pos['attempts'],neg['attempts']))
assert before=={str(p):sha(p) for p in files}
output={'complete':True,'synthetic_feedback_only':True,'real_generation':False,'real_loader_execution':False,'external_processes_intercepted':True,'production_code_unchanged':True,'files_sha256':before,'reference_datasets':reference,'results':results,'opposite_extra_error_probe_same_refinement_and_condition':same,'interpretation':'Current general runner/refiner/backend command path does not select prevalidated phase datasets. Existing generic structural/motion controls are not equivalent to the validated native phase edits. signed_lateral_error_px is an exploratory unsupported probe, not a current production metric. Synthetic trajectories are not measured experiment results.'}
(OUT/'results.json').write_text(json.dumps(output,indent=2)+'\n');print('CLOSED_LOOP_NATIVE_AUDIT_COMPLETE',same,OUT,flush=True)
