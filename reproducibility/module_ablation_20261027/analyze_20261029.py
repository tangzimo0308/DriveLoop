"""Verify artifacts and summarize all fixed assignments; no inference or edits to runs."""
import collections,csv,itertools,json,math,random,statistics,sys,time
from pathlib import Path
ROOT=Path('/home/lesliton/projects/DriveLoop');RUN=ROOT/'outputs/revision_20261027_module_rollout'
sys.path.insert(0,str(RUN))
import run_module_ablation as m
OUT=ROOT/'outputs/revision_20261029_module_analysis';OUT.mkdir(exist_ok=True)
p=m.unseal(RUN/'protocol.json');assert m.digest(p)=='de8bed9a849a2c19d3de739aee4b90b730050a041b2f2ee73b8720fb63c7c61d'
f=m.unseal(m.ref.OUT/'freeze.json');assert m.digest(f)==m.EXPECTED
cache={};hashed_bytes=0

def verify(files):
 global hashed_bytes
 for path,h in files.items():
  q=Path(path);s=q.stat();key=(s.st_size,s.st_mtime_ns,s.st_ctime_ns)
  if path not in cache or cache[path][0]!=key:
   cache[path]=(key,m.sha(q));hashed_bytes+=s.st_size
  assert cache[path][1]==h,'Artifact changed: '+path
m.ref.verify_files=verify
verify(p['files']);verify({q:f['files'][q] for q in f['code_inventory']})
pre=m.read(m.ref.SOURCE/'source_preflight.json');verify(pre['source_hashes'])
manifests={c['case_id']:m.read(c['manifest']) for c in pre['cases']}
plan=m.read(m.PRE/'request_plan.json');observed=m.read(m.PRE/'observed_inputs.json')
known={}
for k,r in plan['unique_requests'].items():
 known.setdefault((r['case_id'],r['backend_mode']),{})[r['request_sha256']]=observed[k]['input_audit']
methods=('full','without_synthetic','without_size');rows=[];calls=[];counts=collections.Counter();evidence={}
for gi,gp in enumerate(p['groups']):
 case,bank=gp['case_id'],gp['bank'];manifest=manifests[case]
 old=m.ref.OUT/'groups'/case/str(bank);new=RUN/'groups'/case/str(bank)
 noise={};full_latents=None;full_result=None
 for mode in methods:
  arm=(old/'feedback') if mode=='full' else new/mode
  op=arm/'outcome.json';saved=m.unseal(op);evidence[str(op)]=m.sha(op)
  assert saved.get('freeze_sha256')==m.EXPECTED if mode=='full' else saved['protocol_sha256']==m.digest(p)
  verify(saved['files']);result=saved['result'];history=result['attempt_history']
  assert saved['completion']['complete'] and len(history)==4
  assert [a['iteration'] for a in history]==list(range(4))
  assert saved['completion']['attempt_count']==4 and saved['completion']['termination']=='candidate_cap_reached'
  if mode=='full':full_result=result
  bk=m.BACKEND_VERSION if mode=='without_synthetic' else 'frozen_reference'
  ka=known[(case,bk)];source=next(iter(ka.values()))
  for i,a in enumerate(history):
   rp=arm/'slots'/f'slot_{i:02d}'/'receipt.json';receipt=m.unseal(rp);verify(receipt['files']);evidence[str(rp)]=m.sha(rp)
   rg=receipt['generation'];ag=a['generation']
   assert set(rg)==set(ag)
   assert all(rg[k]==ag[k] for k in rg if k!='metadata')
   assert all(ag['metadata'].get(k)==v for k,v in rg['metadata'].items())
   extras=set(ag['metadata'])-set(rg['metadata'])
   assert extras.issubset({'control_coverage','control_visibility','dd2_condition','long_tail_condition_plan',
       'perception_baseline_video_exists','perception_baseline_video_resolved','perception_detector_runtime',
       'scene_specification','source_selection','task_utility'}),extras
   # The runner/evaluator append annotations after the generation receipt is sealed.
   # Existing generation fields must still agree exactly.
   assert ag['metadata']['task_utility']['J']==a['evaluation']['metrics']['J']
   dest=Path(receipt['try_directory']);generation=m.Generation(**receipt['generation'])
   req=m.read(dest/'started.json')['request'];actual=m.ref.validate_capture(dest)
   if mode=='full':noise[i]=actual
   else:
    proof_path=new/'noise_pairs'/mode/f'{i:02d}.json'
    proof=m.unseal(proof_path);assert proof['noise']==actual==noise[i] and proof['matched_frozen_full']
    evidence[str(proof_path)]=m.sha(proof_path)
    m.validator(manifest,bank,mode,new,ka,source,noise)(generation,dest)
    counts['new_noise_pairs_verified']+=1
    if mode=='without_synthetic':
     audit=generation.metadata['dd2_override_json']['audit']['synthetic_intervention']
     counts['no_synthetic_slots_retaining_real_tracks']+=audit['retained_real_count']>0
    else:assert req['condition'].get('structural_escalation',{}).get('size_scale',1)==1
   if i==0:
    sig=m.read(dest/'capture/hook_state.json')['final_latents.pt']
    if mode=='full':full_latents=sig
    else:counts[mode+'_slot0_final_latent_equal_full']+=sig==full_latents
   metric=a['evaluation']['metrics']
   calls.append({'case_id':case,'scene_token':manifest['scene_token'],'bank':bank,'method':mode,'iteration':i,
                 'request_sha256':m.digest(req),'request_previously_audited':m.digest(req) in ka,
                 'J':metric['J'],'S_perc':metric['S_perc'],'S_ctrl':metric['S_ctrl'],
                 'direction_consistent':metric.get('maneuver_direction_consistent'),
                 'receipt':str(rp),'receipt_sha256':m.sha(rp)})
   if mode!='full':counts[mode+'_unseen_input_requests']+=m.digest(req) not in ka
  returned=result['returned_generation']['iteration'];assert result['returned_evaluation']==history[returned]['evaluation']
  assert saved['completion']['returned_iteration']==returned
  metric=result['returned_evaluation']['metrics']
  rows.append({'case_id':case,'scene_token':manifest['scene_token'],'bank':bank,'method':mode,
     'return_status':result['return_status'],'accepted_proxy':int(result['return_status']=='accepted'),
     'returned_iteration':returned,'J':metric['J'],'S_perc':metric['S_perc'],'S_ctrl':metric['S_ctrl'],
     'S_intent':metric['S_intent'],'direction_consistent':metric.get('maneuver_direction_consistent'),
     'Q_cov':metric.get('Q_cov'),'outcome_sha256':m.sha(op)})
 support=m.unseal(old/'support/slots/slot_00/receipt.json');verify(support['files'])
 for item in m.read(new/'reference_reuse.json'):
  assert m.sha(item['receipt'])==item['sha256']
 print('GROUP_VERIFIED',gi+1,'of',len(p['groups']),case,bank,flush=True)
ledger=m.read(RUN/'ledger.json')
assert ledger['complete_new_arms']==72
assert ledger['new_completed_receipts']==ledger['new_generation_calls_started']==288
assert ledger['evaluation_calls_completed']==ledger['evaluation_calls_started']==288
assert ledger['extra_retry_starts']==0
# Verify all evaluation receipts, including exact agreement with outcome scores.
for group in p['groups']:
 for mode in m.METHODS:
  arm=RUN/'groups'/group['case_id']/str(group['bank'])/mode
  saved=m.unseal(arm/'outcome.json');history=saved['result']['attempt_history']
  records=sorted(arm.glob('history_*/evaluation_calls/*.json'))
  assert len(records)==4
  for path in records:
   r=m.read(path);assert r['complete']
   raw=r['evaluation'];final=history[r['iteration']]['evaluation']
   assert raw['diagnosis']==final['diagnosis']
   assert all(final['metrics'].get(k)==v for k,v in raw['metrics'].items())
   assert set(final['metrics'])-set(raw['metrics'])=={'J','S_perc','S_ctrl','S_intent'}
   assert raw['score']==final['metrics']['S_perc']
   assert final['score']==final['metrics']['J']
   mm=final['metrics'];assert abs(mm['J']-(.5*mm['S_perc']+.3*mm['S_ctrl']+.2*mm['S_intent']))<=1.1e-6
for name,data in [('selected_results.csv',rows),('all_candidates.csv',calls)]:
 with (OUT/name).open('w') as stream:
  w=csv.DictWriter(stream,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
summary={}
for method in methods:
 r=[a for a in rows if a['method']==method]
 assert len(r)==36
 summary[method]={'n_scene_seed_pairs':len(r),'status_counts':dict(collections.Counter(a['return_status'] for a in r)),
                  'accepted_proxy_count':sum(a['accepted_proxy'] for a in r),
                  'means':{k:statistics.mean(a[k] for a in r) for k in ('J','S_perc','S_ctrl','S_intent')},
                  'direction_unavailable_count':sum(a['direction_consistent'] is None for a in r)}
# Pair at scene+seed, aggregate by unique source scene before uncertainty calculation.
scenes=sorted({r['scene_token'] for r in rows});assert len(scenes)==12
index={(r['method'],r['case_id'],r['bank']):r for r in rows}
def percentile(x,p):
 s=sorted(x);q=(len(s)-1)*p;lo=int(q);hi=math.ceil(q)
 return s[lo]+(s[hi]-s[lo])*(q-lo)
paired=[]
for variant in methods[1:]:
 for metric in ('accepted_proxy','J','S_perc'):
  scene_deltas=[];deltas=[]
  for scene in scenes:
   group=[r for r in rows if r['method']=='full' and r['scene_token']==scene]
   assert len(group)==3
   dd=[r[metric]-index[(variant,r['case_id'],r['bank'])][metric] for r in group]
   deltas+=dd;scene_deltas.append(statistics.mean(dd))
  rng=random.Random(20261029)
  bootstrap=[statistics.mean(rng.choices(scene_deltas,k=len(scenes))) for _ in range(20000)]
  estimate=statistics.mean(scene_deltas)
  flips=[statistics.mean(s*d for s,d in zip(signs,scene_deltas)) for signs in itertools.product((-1,1),repeat=12)]
  paired.append({'contrast':'full minus '+variant,'metric':metric,'mean_difference':estimate,
     'cluster_bootstrap_95_percentile_CI':[percentile(bootstrap,.025),percentile(bootstrap,.975)],
     'exploratory_two_sided_scene_sign_flip_p':sum(abs(x)>=abs(estimate)-1e-12 for x in flips)/len(flips),
     'scene_seed_wins_ties_losses':[sum(x>1e-12 for x in deltas),sum(abs(x)<=1e-12 for x in deltas),sum(x< -1e-12 for x in deltas)],
     'source_scene_mean_differences':scene_deltas})
report={'date':'2026-10-29','protocol_sha256':m.digest(p),'verification_passed':True,
        'new_generation_count':288,'new_evaluation_count':288,'retry_starts':0,
        'source_scene_count':12,'banks_per_scene':3,'candidate_cap':4,
        'integrity_verified_file_count':len(cache),'integrity_hashed_bytes':hashed_bytes,
        'runtime_checks':dict(counts),'summary':summary,'paired_exploratory_analysis':paired,
        'generation_completed_seconds':ledger['generation_completed_wall_seconds'],
        'uncertainty_method':'20,000 paired source-scene bootstrap samples, seed20261029; 12 source scenes, 3 matched seed banks each. Exact 4096 scene-level sign flips reported descriptively, unadjusted for multiple comparisons. Not a randomized-treatment inference.',
        'scope':'Post-hoc diagnostic on previously inspected scenes. Return-status acceptance uses automatic proxies, not human-confirmed semantics. No new human review yet.',
        'model_assets_rehashed_in_this_analysis':False,
        'integrity_scope':'Protocol, frozen code/source inputs, outcome/evaluation records, new/reference generation artifacts and noise pairing. Model assets were checked by generation launcher, not rehashed here.'}
m.atomic_json(OUT/'analysis.json',report);m.atomic_json(OUT/'evidence_manifest.json',evidence)
print('VERIFICATION_AND_ANALYSIS_COMPLETE',json.dumps({'summary':summary,'runtime_checks':dict(counts)}),flush=True)
