"""Real loader, synthetic diagnostic inputs, no diffusion. Positive/negative/missing paths."""
import json,os,time
from pathlib import Path
from dataclasses import asdict
from driveloop.native_phase import NativePhaseRegistry,NativePhaseBackend,NativePhaseEvaluator,NativePhaseRefiner,NativePhaseRunner,NativePhaseMeasurement,digest
from driveloop.schema import DriveLoopConfig
from scripts.run_native_bus_control import setup_env,make_backend
ROOT=Path('/home/lesliton/projects/DriveLoop');PRE=ROOT/'outputs/revision_20261008_candidate1583_inputs/preflight.json'
OUT=ROOT/'outputs/revision_20261009_native_phase_loader_audit';OUT.mkdir(exist_ok=False)
setup_env();start=time.monotonic();registry=NativePhaseRegistry(PRE,'997384cdb119b1679abc84e48276da54f38e3141d297271e50e192bd499d9af2')
protocol={'type':'real_loader_with_synthetic_feedback','generation_run':False,'synthetic_values_are_not_model_results':True,'test_tolerance_px':1,'tolerance_scope':'Software branch test only, not a paper success criterion','registry_sha256':registry.fingerprint,'core_file_hashes':json.loads((ROOT/'outputs/revision_20261009_closed_loop_native_audit/results.json').read_text())['files_sha256'],'new_module_sha256':digest(ROOT/'driveloop/native_phase.py'),'script_sha256':digest(__file__),'checks':['positive to advanced','negative to delayed','missing stops before second loader','approved tensor hashes','cached prompt/image/map and first-frame proof inherited from pinned preflight','same scene and instance','core files unchanged']}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');cases=[]
def factory(manifest,path,audit_only):
 assert audit_only;return make_backend(manifest,path,audit_only=True)
for name,target_arm,missing in [('positive','advanced',False),('negative','delayed',False),('missing','advanced',True)]:
 dest=OUT/name;backend=NativePhaseBackend(registry,factory,dest/'backend',audit_only=True)
 def measure(g):
  arm=g.metadata['native_phase_contract']['arm'];vals=(None,0,0,0) if missing else (registry.means[arm],)*4
  return NativePhaseMeasurement(registry.scene,registry.instance,g.iteration,arm,vals,{'synthetic_software_probe':True,'not_video_measurement':True},synthetic=True)
 runner=NativePhaseRunner(backend,NativePhaseEvaluator(measure,allow_synthetic=True),NativePhaseRefiner(registry),DriveLoopConfig(max_iterations=3,target_score=0,output_dir=dest/'history'))
 result=runner.run(registry.request(registry.means[target_arm],1));attempts=result.attempt_history;actions=[a.generation.metadata['native_phase_route']['arm'] for a in attempts]
 assert actions==(['original'] if missing else ['original',target_arm]),actions
 assert result.return_status==('perception_unmeasurable' if missing else 'accepted')
 for a in attempts:
  assert a.generation.metadata['dd2_audit_only'] and 'video' not in a.generation.artifacts
  assert a.generation.metadata['dd2_prompt'] is None and a.generation.metadata['dd2_override_json']['available'] is False
 row={'name':name,'actions':actions,'return_status':result.return_status,'software_probe_only':True,'attempts':[asdict(a) for a in attempts]};cases.append(row)
 (OUT/'partial_results.json').write_text(json.dumps(cases,indent=2,default=str)+'\n');print('REAL_LOADER_PATH_PASS',name,actions,result.return_status,flush=True)
for p,h in protocol['core_file_hashes'].items():assert digest(p)==h,p
assert not list(OUT.rglob('*.mp4'))
(OUT/'summary.json').write_text(json.dumps({'passed':True,'actual_loader_calls':sum(len(c['actions']) for c in cases),'generation_run':False,'synthetic_feedback_only':True,'production_core_unchanged':True,'cases':cases,'seconds':time.monotonic()-start},indent=2,default=str)+'\n');print('NATIVE_PHASE_REAL_LOADER_AUDIT_COMPLETE',OUT,flush=True)
