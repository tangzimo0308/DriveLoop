"""Verify saved pilot evidence without new generation or detector inference."""
import json, statistics, time
from pathlib import Path
from types import SimpleNamespace
from driveloop.native_phase import NativePhaseRegistry,digest
from driveloop.native_phase_png import associate
from driveloop.phase_input_receipt import PhaseInputReceipt
from driveloop.phase_amplitude import propose_phase
from scripts.run_dynamic_phase_pilot import PRE,PRE_SHA,ROOT,RANGE,verify
from scripts.run_native_matched_capture import validate_hooks
RUN=ROOT/'outputs/dynamic_amplitude_20261014_222304_43184'
OUT=ROOT/'outputs/revision_20261014_amplitude_results_audit'
OUT.mkdir(exist_ok=False);start=time.monotonic()
read=lambda p:json.loads(Path(p).read_text())
s=read(RUN/'summary.json');spec=read(RUN/'protocol.json')['specification']
r=NativePhaseRegistry(PRE,PRE_SHA);verify(spec,r)
curve=read(RANGE/'results.json')['rows']
assert s['complete'] and s['all_methods_completed_budget'] and s['all_attempts_measurable']
ledger=read(RUN/'generation_ledger.json');assert ledger==s['generation_ledger'] and len(ledger)==21
admissions={}
for a in s['admissions']:
 assert a['status']=='admitted'
 admissions[a['phase_s']]=PhaseInputReceipt(a['receipt'],a['receipt_sha256'],a['phase_s'],r)
assert len(admissions)==8 and s['actual_admission_loader_calls']==16
receipts={};groups={};refs={}
for p in RUN.rglob('capture_receipt.json'):
 row=read(p);g=row['generation'];q=g['metadata']['native_phase_contract'];i=g['iteration']
 assert digest(g['artifacts']['video'])==row['video_sha256']
 h=validate_hooks(Path(row['capture_dir']))
 sig=tuple(h['files'][f'generated_front/frame{j}.png'] for j in range(8))
 groups.setdefault((i,q['phase_s']),[]).append(sig)
 receipts[str(p)]=(row,h)
 if '/references/' in str(p):refs[i]=(row,h)
assert len(receipts)==21 and set(refs)=={0,1,2} and all(len(set(v))==1 for v in groups.values())
rows=[];unique=0;maxdiff=0
for m in s['methods']:
 direction,name=m['direction'],m['method'];result=read(RUN/direction/name/'result.json');attempts=result['attempt_history'];assert len(attempts)==3
 derived=[]
 for i,a in enumerate(attempts):
  g=a['generation'];f=a['evaluation']['metrics']['native_phase_feedback'];q=g['metadata']['native_phase_contract'];phase=q['phase_s']
  assert q==f['contract']==a['request']['condition']['native_phase']
  assert q['target_dx_px']==spec['targets_px'][direction] and q['tolerance_px']==1
  assert g['iteration']==i and not f['synthetic'] and f['status']=='measured'
  admissions[phase].verify_runtime(SimpleNamespace(metadata=g['metadata'],artifacts=g['artifacts']),audit_only=False)
  row,h=receipts[g['artifacts']['native_phase_receipt']];orig,oh=refs[i]
  for k in ('initial_noise','image_latent_before'):
   assert row['latent_audit'][k]['sha256']==orig['latent_audit'][k]['sha256']
  pr=f['provenance'];assert pr['late_indices']==[4,5,6,7] and pr['detector']['confidence']==.5 and pr['detector']['unique_bus_iou']==.3
  for role,hh in [('current',h),('reference',oh)]:
   clip=pr[role];assert clip['seed_offset']==i and clip['scene_token']==r.scene and clip['instance_token']==r.instance
   assert digest(clip['evidence_path'])==clip['evidence_sha256']
   assert len(clip['frames'])==len(pr[role+'_detections'])==8
   for j,(frame,det) in enumerate(zip(clip['frames'],pr[role+'_detections'])):
    assert frame['index']==det['frame']==j and frame['sha256']==det['png_sha256']==hh['files'][f'generated_front/frame{j}.png']
    u=associate(det['raw'],frame['projected_box']);assert u['association_count']==det['association_count']==1 and u['target']==det['target']
    if role=='current':unique+=1
  dx=[]
  for j in range(4,8):
   b=pr['current_detections'][j]['target']['box'];o=pr['reference_detections'][j]['target']['box'];dx.append((b[0]+b[2]-o[0]-o[2])/2)
  observed=statistics.mean(dx);error=q['target_dx_px']-observed
  diff=max(abs(observed-f['observed_dx_px']),abs(error-f['error_px']),max(abs(x-y) for x,y in zip(dx,f['late_deltas'])))
  maxdiff=max(maxdiff,diff);assert diff<1e-9
  assert abs(m['attempts'][i]['absolute_error_px']-abs(error))<1e-9 and m['attempts'][i]['phase_s']==phase
  if i<2:
   nextphase=attempts[i+1]['request']['condition']['native_phase']['phase_s']
   assert a['refinement']['condition']==attempts[i+1]['request']['condition']
   if name=='residual_feedback':assert propose_phase(curve,phase,q['target_dx_px'],dx)['proposed_phase_s']==nextphase
  if name=='direct_target':assert phase==spec['initial_phase_s'][direction]
  if name=='fixed_amplitude_search':assert phase==[.25,.125,.375][i]*(1 if direction=='delayed' else -1)
  derived.append({'iteration':i,'phase_s':phase,'observed_dx_px':observed,'absolute_error_px':abs(error)})
 best=min(derived,key=lambda x:x['absolute_error_px']);assert best['iteration']==m['best_measured']['iteration']==result['best_generation']['iteration']
 rows.append({'direction':direction,'method':name,'best_error_px':best['absolute_error_px'],'attempts':derived})
assert unique==144 and sum(x['logical_total_renders'] for x in s['methods'])==36
out={'passed':True,'source_run':str(RUN),'generation_run':False,'detector_inference_run':False,'summary_sha256':digest(RUN/'summary.json'),'protocol_sha256':digest(RUN/'protocol.json'),'capture_receipts_verified':len(receipts),'front_frames_verified':len(receipts)*8,'candidate_unique_associations':unique,'max_recomputed_error_difference':maxdiff,'distinct_seed_phase_outputs':len(groups),'same_seed_phase_outputs_identical':True,'exact_admissions_verified':len(admissions),'admission_loader_calls':16,'actual_renders':21,'logical_renders':36,'elapsed_run_seconds':s['elapsed_seconds'],'methods':rows,'audit_seconds':time.monotonic()-start,'limits':['Saved detector outputs reassociated, detector not rerun.','Single inspected development scene, no independent holdout or physical ground truth.','Subpixel best-of-three improvements require independent evaluation and measurement uncertainty assessment.','Equal generation budgets, not equal total compute; admission cache is shared.']}
(OUT/'results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
