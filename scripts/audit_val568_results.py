"""Read-only verification of saved evidence; no generation or detector inference."""
import json,time
from pathlib import Path
from types import SimpleNamespace
from driveloop.native_phase import NativePhaseRegistry,digest
from driveloop.native_phase_png import associate
from driveloop.phase_input_receipt import PhaseInputReceipt
from scripts.run_val568_pilot import PRE,PRE_SHA,ROOT,verify
from scripts.run_native_matched_capture import validate_hooks
RUN=ROOT/'outputs/val568_pilot_20261019_143140_28678'
OUT=ROOT/'outputs/revision_20261019_val568_audit'
OUT.mkdir(exist_ok=False);start=time.monotonic()
read=lambda p:json.loads(Path(p).read_text())
s=read(RUN/'summary.json');spec=read(RUN/'protocol.json')['specification']
r=NativePhaseRegistry(PRE,PRE_SHA);verify(spec,r)
assert s['complete'] and not s['all_methods_completed_budget'] and not s['all_attempts_measurable']
assert read(RUN/'generation_ledger.json')==s['generation_ledger'] and len(s['generation_ledger'])==9
admissions={}
for a in s['admissions']:
 assert a['status']=='admitted'
 admissions[a['phase_s']]=PhaseInputReceipt(a['receipt'],a['receipt_sha256'],a['phase_s'],r)
assert len(admissions)==6 and s['actual_admission_loader_calls']==s['preparation_loader_calls']==12
receipts={};groups={};refs={}
for p in RUN.rglob('capture_receipt.json'):
 row=read(p);g=row['generation'];q=g['metadata']['native_phase_contract'];i=g['iteration']
 assert digest(g['artifacts']['video'])==row['video_sha256']
 h=validate_hooks(Path(row['capture_dir']))
 sig=tuple(h['files'][f'generated_front/frame{j}.png'] for j in range(8))
 groups.setdefault((i,q['phase_s']),[]).append(sig)
 receipts[str(p)]=(row,h)
 if '/references/' in str(p):refs[i]=(row,h)
assert len(receipts)==9 and set(refs)=={0,1,2} and all(len(set(v))==1 for v in groups.values())
rows=[]
for m in s['methods']:
 direction,name=m['direction'],m['method'];result=read(RUN/direction/name/'result.json');attempts=result['attempt_history'];assert len(attempts)==1
 a=attempts[0];g=a['generation'];f=a['evaluation']['metrics']['native_phase_feedback'];q=g['metadata']['native_phase_contract'];phase=q['phase_s']
 assert q==f['contract']==a['request']['condition']['native_phase'] and phase==spec['initial_phase_s'][direction]
 assert q['target_dx_px']==spec['targets_px'][direction] and q['tolerance_px']==1
 assert g['iteration']==0 and not f['synthetic'] and f['status']=='unmeasured'
 assert f['error_px'] is None and f['observed_dx_px'] is None and m['best_measured'] is None
 admissions[phase].verify_runtime(SimpleNamespace(metadata=g['metadata'],artifacts=g['artifacts']),audit_only=False)
 row,h=receipts[g['artifacts']['native_phase_receipt']];orig,oh=refs[0]
 for k in ('initial_noise','image_latent_before'):
  assert row['latent_audit'][k]['sha256']==orig['latent_audit'][k]['sha256']
 pr=f['provenance'];assert pr['late_indices']==[4,5,6,7] and pr['detector']['confidence']==.5 and pr['detector']['unique_bus_iou']==.3
 counts={};raw={}
 for role,hh in [('current',h),('reference',oh)]:
  clip=pr[role];assert clip['seed_offset']==0 and clip['scene_token']==r.scene and clip['instance_token']==r.instance
  assert digest(clip['evidence_path'])==clip['evidence_sha256']
  assert len(clip['frames'])==len(pr[role+'_detections'])==8
  for j,(frame,det) in enumerate(zip(clip['frames'],pr[role+'_detections'])):
   assert frame['index']==det['frame']==j and frame['sha256']==det['png_sha256']==hh['files'][f'generated_front/frame{j}.png']
   u=associate(det['raw'],frame['projected_box']);assert u['association_count']==det['association_count'] and u['target']==det['target']
  counts[role]=[d['association_count'] for d in pr[role+'_detections']]
  raw[role]=[[{'label':v['label'],'score':v['score']} for v in d['raw']] for d in pr[role+'_detections']]
 dx=[]
 for j in range(4,8):
  b=pr['current_detections'][j]['target'];o=pr['reference_detections'][j]['target']
  dx.append(None if b is None or o is None else (b['box'][0]+b['box'][2]-o['box'][0]-o['box'][2])/2)
 assert dx==f['late_deltas'] and any(v is None for v in dx)
 rows.append({'direction':direction,'method':name,'return_status':result['return_status'],'association_counts':counts,'raw_labels_scores':raw,'late_deltas':dx,'refinement':a.get('refinement'),'best_measured':None})
assert sum(x['logical_total_renders'] for x in s['methods'])==24
out={'passed':True,'source_run':str(RUN),'generation_run':False,'detector_inference_run':False,'summary_sha256':digest(RUN/'summary.json'),'protocol_sha256':digest(RUN/'protocol.json'),'capture_receipts_verified':len(receipts),'front_frames_verified':len(receipts)*8,'distinct_seed_phase_outputs':len(groups),'same_seed_phase_outputs_identical':True,'exact_admissions_verified':len(admissions),'admission_loader_calls':12,'actual_renders':9,'logical_renders':24,'elapsed_run_seconds':s['elapsed_seconds'],'methods':rows,'audit_seconds':time.monotonic()-start,'conclusion':'All six method-direction runs stopped after the shared-policy first candidate because the fixed complete-late-frame bus measurement was unavailable. No comparative feedback benefit can be estimated.','limits':['Saved detector outputs reassociated; detector not rerun.','No physical truth or manual class judgement established.','Six method failures share two initial outputs and one reference, not six independent failures.']}
(OUT/'results.json').write_text(json.dumps(out,indent=2)+'\n')
print('AUDIT_PASS',len(receipts),'captures;',len(groups),'distinct seed/phase outputs')
for x in rows:print(x['direction'],x['method'],x['association_counts'],x['late_deltas'],x['return_status'])
