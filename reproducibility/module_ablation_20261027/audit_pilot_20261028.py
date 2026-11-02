"""Read-only pilot verification; no model inference or outcome modification."""
import copy,json,sys,shutil
from pathlib import Path
ROOT=Path('/home/lesliton/projects/DriveLoop')
OUT=ROOT/'outputs/revision_20261027_module_rollout'
sys.path.insert(0,str(OUT))
import run_module_ablation as m
p=m.unseal(OUT/'protocol.json');m.ref.verify_files(p['files'])
f=m.unseal(m.ref.OUT/'freeze.json');assert m.digest(f)==m.EXPECTED
m.ref.verify_files({q:f['files'][q] for q in f['code_inventory']})
pre=m.read(m.ref.SOURCE/'source_preflight.json')
plan=m.read(m.PRE/'request_plan.json');observed=m.read(m.PRE/'observed_inputs.json')
group=p['groups'][0];case,bank=group['case_id'],group['bank']
new=OUT/'groups'/case/str(bank);old=m.ref.OUT/'groups'/case/str(bank)
manifest=m.read(next(c['manifest'] for c in pre['cases'] if c['case_id']==case))
refs={};noise={};rows=[];total_bytes=0
for i in range(4):
 r=m.unseal(old/'feedback/slots'/f'slot_{i:02d}'/'receipt.json')
 m.ref.verify_files(r['files']);refs[i]=r
 noise[i]=m.ref.validate_capture(Path(r['try_directory']))
full=m.unseal(old/'feedback/outcome.json');m.ref.verify_files(full['files'])
for mode in m.METHODS:
 arm=new/mode;result=m.unseal(arm/'outcome.json')
 assert result['protocol_sha256']==m.digest(p)
 m.ref.verify_files(result['files'])
 assert result['completion']['complete'] and result['completion']['attempt_count']==4
 backend_mode=m.BACKEND_VERSION if mode=='without_synthetic' else 'frozen_reference'
 known={r['request_sha256']:observed[k]['input_audit'] for k,r in plan['unique_requests'].items()
        if r['case_id']==case and r['backend_mode']==backend_mode}
 validate=m.validator(manifest,bank,mode,new,known,next(iter(known.values())),noise)
 checks=[]
 for i in range(4):
  receipt=m.unseal(arm/'slots'/f'slot_{i:02d}'/'receipt.json')
  m.ref.verify_files(receipt['files']);dest=Path(receipt['try_directory'])
  generation=m.Generation(**receipt['generation']);validate(generation,dest)
  proof=m.unseal(new/'noise_pairs'/mode/f'{i:02d}.json')
  assert proof['noise']==noise[i] and proof['matched_frozen_full']
  req=m.read(dest/'started.json')['request']
  if mode=='without_synthetic':
   boxes=generation.metadata['dd2_override_json']['boxes3d']
   assert not any(boxes[k] for k in ('append','per_frame_append','per_frame_append_ego'))
  assert req['condition'].get('structural_escalation',{}).get('size_scale',1)==1 if mode=='without_size' else True
  total_bytes+=sum(q.stat().st_size for q in dest.rglob('*') if q.is_file())
  checks.append({'iteration':i,'noise_matched':True,'request_previously_input_audited':m.digest(req) in known,
                 'slot0_final_latent_tensor_equal_to_reference':m.read(dest/'capture/hook_state.json')['final_latents.pt']==m.read(Path(refs[i]['try_directory'])/'capture/hook_state.json')['final_latents.pt'] if i==0 else None})
 r=result['result'];e=r['returned_evaluation']
 rows.append({'mode':mode,'completion':result['completion'],'selected_metrics':{k:e['metrics'].get(k) for k in
              ('J','S_perc','S_ctrl','S_intent','maneuver_direction_consistent','Q_cov')},'selected_diagnosis':e['diagnosis'],
              'checks':checks})
for item in m.read(new/'reference_reuse.json'):
 assert m.sha(item['receipt'])==item['sha256']
 receipt=m.unseal(item['receipt']);m.ref.verify_files(receipt['files'])
ledger=m.read(OUT/'ledger.json')
assert ledger['new_completed_receipts']==ledger['new_generation_calls_started']==8
assert ledger['evaluation_calls_completed']==ledger['evaluation_calls_started']==8
assert ledger['extra_retry_starts']==0
r=full['result'];e=r['returned_evaluation']
report={'case':case,'bank':bank,'protocol_sha256':m.digest(p),'verification_passed':True,
        'completed_arms':2,'completed_generations':8,'completed_evaluations':8,'noise_pairs_verified':8,'retry_starts':0,
        'full_reference':{'return_status':r['return_status'],'returned_iteration':r['returned_generation']['iteration'],
                          'selected_metrics':{k:e['metrics'].get(k) for k in ('J','S_perc','S_ctrl','S_intent','maneuver_direction_consistent')}},
        'variants':rows,'remaining_groups':35,'remaining_arms':70,'remaining_candidate_upper_bound':280,
        'pilot_generation_seconds':ledger['generation_completed_wall_seconds'],
        'pilot_artifact_bytes':total_bytes,'free_disk_bytes':shutil.disk_usage(OUT).free,
        'claim':'One scene-bank execution check, not evidence of aggregate or semantic superiority. No human labels collected here.'}
path=ROOT/'outputs/revision_20261028_module_pilot_audit';path.mkdir(exist_ok=True)
m.atomic_json(path/'verification.json',report)
print(json.dumps(report,indent=2))
