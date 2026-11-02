"""Read-only independent audit of completed development pilot; no new inference."""
import json, statistics, time
from pathlib import Path
from driveloop.native_phase import NativePhaseRegistry, digest, NativePhaseRefiner
from driveloop.native_phase_png import associate
from driveloop.schema import DriveLoopRequest, Evaluation, Diagnosis
from scripts.run_native_phase_online import PRE,PRE_SHA,ROOT,plans_for,verify
from scripts.run_native_matched_capture import validate_hooks

RUN=ROOT/'outputs/native_phase_online_20261013_081629_15056'
OUT=ROOT/'outputs/revision_20261013_online_results_audit'
OUT.mkdir(exist_ok=False);start=time.monotonic()
s=json.loads((RUN/'summary.json').read_text());protocol=json.loads((RUN/'protocol.json').read_text());spec=protocol['specification']
r=NativePhaseRegistry(PRE,PRE_SHA);verify(spec,r);plans=plans_for(r)
assert s['complete'] and s['all_methods_completed_budget'] and s['all_attempts_measurable']
ledger=json.loads((RUN/'generation_ledger.json').read_text());assert ledger==s['generation_ledger'] and len(ledger)==27
assert len({x['receipt'] for x in ledger})==27
receipts={}; groups={}; frames=0
for event in ledger:
 p=Path(event['receipt']);row=json.loads(p.read_text());assert (row['arm'],row['seed_offset'])==(event['arm'],event['seed_offset'])
 assert digest(row['generation']['artifacts']['video'])==row['video_sha256']
 h=validate_hooks(Path(row['capture_dir']));frames+=8
 sig=tuple(h['files'][f'generated_front/frame{i}.png'] for i in range(8))
 key=(row['seed_offset'],row['arm']);groups.setdefault(key,[]).append(sig)
 receipts[str(p)]=(row,h)
assert len(groups)==9 and all(len(set(v))==1 for v in groups.values())
refs={row['seed_offset']: (row,h) for path,(row,h) in receipts.items() if '/references/' in path}
assert set(refs)=={0,1,2}
results=[];max_recompute_difference=0;all_unique=0;notes=[]
for method in s['methods']:
 target=method['target_arm'];name=method['method']
 result=json.loads((RUN/target/name/'result.json').read_text());attempts=result['attempt_history']
 assert len(attempts)==3 and [a['iteration'] for a in attempts]==[0,1,2]
 derived=[]
 for i,a in enumerate(attempts):
  g=a['generation'];f=a['evaluation']['metrics']['native_phase_feedback'];q=g['metadata']['native_phase_contract'];arm=q['arm']
  assert q==f['contract']==a['request']['condition']['native_phase']
  assert q['registry_sha256']==r.fingerprint and q['target_dx_px']==spec['targets'][target] and q['tolerance_px']==1
  assert not f['synthetic'] and not g['metadata']['dd2_audit_only'] and g['metadata']['native_phase_route']['runtime_inputs_match']
  path=g['artifacts']['native_phase_receipt'];row,h=receipts[path]
  assert row['seed_offset']==i and row['arm']==arm
  orig,oh=refs[i]
  for k in ('initial_noise','image_latent_before'):
   assert row['latent_audit'][k]['sha256']==orig['latent_audit'][k]['sha256']
  for k in ('prompt_embed','img_cond','grounding_downsampler_input','box_downsampler_input'):
   assert g['metadata']['dd2_runtime_input_audit'][k]['sha256']==r.case['runtime_audits'][arm][k]['sha256']
  pr=f['provenance'];assert pr['late_indices']==[4,5,6,7] and pr['detector']['confidence']==.5 and pr['detector']['unique_bus_iou']==.3
  for role,hh,expected_arm in [('current',h,arm),('reference',oh,'original')]:
   clip=pr[role];assert clip['seed_offset']==i and clip['arm']==expected_arm
   assert clip['scene_token']==r.scene and clip['instance_token']==r.instance
   assert digest(clip['evidence_path'])==clip['evidence_sha256']
   for j,(frame,det) in enumerate(zip(clip['frames'],pr[role+'_detections'])):
    assert frame['index']==det['frame']==j and frame['sha256']==det['png_sha256']==hh['files'][f'generated_front/frame{j}.png']
    assert frame['projected_box']==list(plans[expected_arm][j])
    unique=associate(det['raw'],frame['projected_box'])
    assert unique['association_count']==det['association_count']==1 and unique['target']==det['target']
    if role=='current':all_unique+=1
  dx=[]
  for j in range(4,8):
   ca=pr['current_detections'][j]['target']['box'];oa=pr['reference_detections'][j]['target']['box']
   dx.append((ca[0]+ca[2]-oa[0]-oa[2])/2)
  observed=statistics.mean(dx);error=q['target_dx_px']-observed
  difference=max(abs(observed-f['observed_dx_px']),abs(error-f['error_px']),max(abs(x-y) for x,y in zip(dx,f['late_deltas'])))
  max_recompute_difference=max(max_recompute_difference,difference);assert difference<=1e-9
  if name=='feedback':
   ev=a['evaluation'];ref=NativePhaseRefiner(r).refine(DriveLoopRequest(**a['request']),Evaluation(ev['score'],ev['metrics'],Diagnosis(**ev['diagnosis'])))
   assert ref.condition==a['refinement']['condition'] and ref.notes==a['refinement']['notes']
  if i<2:assert a['refinement']['condition']==attempts[i+1]['request']['condition']
  m=method['attempts'][i];assert m['arm']==arm and abs(m['absolute_error_px']-abs(error))<=1e-9
  derived.append({'iteration':i,'arm':arm,'observed_dx_px':observed,'absolute_error_px':abs(error),
                  'refinement_notes':a['refinement']['notes']})
 best=min(derived,key=lambda x:x['absolute_error_px'])
 assert best['iteration']==method['best_measured']['iteration'] and method['logical_total_renders']==6
 results.append({'target':target,'method':name,'best_error_px':best['absolute_error_px'],'best_iteration':best['iteration'],
                 'sequence':[a['arm'] for a in derived],'attempts':derived})
assert frames==216 and all_unique==192 and sum(m['logical_total_renders'] for m in s['methods'])==48
comparisons={}
for t in ('delayed','advanced'):
 d=next(x for x in results if x['target']==t and x['method']=='direct_target')
 f=next(x for x in results if x['target']==t and x['method']=='feedback')
 comparisons[t]={'feedback_minus_direct_best_error_px':f['best_error_px']-d['best_error_px'],
                 'feedback_beats_direct':f['best_error_px']<d['best_error_px']}
result={'passed':True,'source_run':str(RUN),'summary_sha256':digest(RUN/'summary.json'),'protocol_sha256':digest(RUN/'protocol.json'),
        'generation_run':False,'detector_inference_run':False,'actual_renders':27,'logical_renders':48,
        'captured_front_frames_verified':frames,'candidate_unique_detection_frames':all_unique,'distinct_seed_action_outputs':9,
        'same_seed_action_png_hashes_identical_across_fresh_renders':True,'max_recomputed_measurement_difference':max_recompute_difference,
        'methods':results,'feedback_vs_direct':comparisons,'sum_generation_seconds':sum(e['seconds'] for e in ledger),
        'audit_seconds':time.monotonic()-start,
        'limits':['Raw detection outputs were reassociated and measurements recomputed; detector was not rerun.',
                  'Original-minus-same-seed-original is zero by construction; it is not an independent unconditional-resampling benchmark.',
                  '27 executed calls produce only 9 unique seed/action clips in this deterministic pilot, not 27 independent samples.',
                  'Single already inspected development scene; no significance or generalized feedback efficacy claim.']}
(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='methods'},indent=2))
for row in results:print(row['target'],row['method'],row['sequence'],row['best_error_px'])
print('ONLINE_RESULTS_AUDIT_COMPLETE',OUT)
