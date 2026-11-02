"""Complete lossless capture for six frozen cases; reuse one exact replay, render five."""
import argparse,copy,json,os,shutil,time,traceback
from dataclasses import asdict
from pathlib import Path
from scripts.run_native_bus_control import ROOT,setup_env,make_backend,REQUEST,assert_native
from scripts.run_native_generation_capture import check_fixed,compare_videos
from scripts.run_native_advance_pair import check_pair
from scripts.dd2_capture_hooks import PIPE,TESTER,sha,save
PRIOR=ROOT/'outputs/native_multiscene_20261007_214304_13198'
REUSE=ROOT/'outputs/native_generation_capture_20261008_131957_51857'
PREFLIGHT=ROOT/'outputs/revision_20261008_matched_capture_preflight'
ARMS=('original','delayed','advanced')
EXPECTED={(seed,arm) for seed in (0,1) for arm in ARMS}
FILES=[ROOT/'scripts'/name for name in ('dd2_capture_hooks.py','dd2_capture_python','run_native_generation_capture.py','run_native_matched_capture.py','run_native_matched_capture.sh')]

def index_cases(rows):
 result={}
 for row in rows:
  assert row['candidate_index']==5596,'Unexpected candidate'
  key=(row['seed_offset'],row['arm']);assert key not in result,('Duplicate case',key);result[key]=row
 assert set(result)==EXPECTED,('Incomplete or extra cases',set(result))
 return result

def validate_hooks(path):
 h=json.loads((path/'hook_state.json').read_text())
 assert set(h['installed'])=={PIPE,TESTER} and len(h['installed'])==2
 assert h['counts']=={'decode':1,'postprocess':1,'composite':1}
 assert h['capture_rng_unchanged']==['before_decode','after_decode','after_postprocess','after_composite']
 assert h['generated_equals_composite_bottom']
 expected={'final_latents.pt','decoded_before_postprocess.pt'}|{f'{folder}/frame{i}.png' for folder in ('generated_panorama','generated_front','composite_before_mp4') for i in range(8)}
 assert set(h['files'])==expected,'Missing or extra capture file inventory'
 for name,digest in h['files'].items():assert sha(path/name)==digest,('Capture file changed',name)
 assert h['decode_settings']=={'num_frames':8,'decode_chunk_size':1,'clip_latents':False,'vae_dtype':'torch.float16','scaling_factor':.18215}
 return h

def validate_row(row,reference,expected):
 assert row['candidate_index']==reference['candidate_index']==5596
 assert (row['seed_offset'],row['arm'])==(reference['seed_offset'],reference['arm'])
 meta=row['generation']['metadata'];assert meta['dd2_override_json']['available'] is False and meta['dd2_prompt'] is None
 for key in ('prompt_embed','img_cond','grounding_downsampler_input','box_downsampler_input'):
  assert meta['dd2_runtime_input_audit'][key]['sha256']==expected[key]['sha256'],key
 assert row['latent_audit']==reference['latent_audit'],'Initial latent audit differs'
 assert row['video_sha256']==reference['video_sha256'],'MP4 SHA differs from original experiment'
 assert row['exact_video_replay'] and row['video_comparison']['all_decoded_pixels_equal']

def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--check-only',action='store_true');args=p.parse_args()
 setup_env();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
 try:
  old_protocol=json.loads((PRIOR/'protocol.json').read_text());pre=old_protocol['preflight'];assert pre['passed'] and not pre['generation_run'];check_fixed(pre)
  case=next(x for x in pre['cases'] if x['candidate_index']==5596);references=index_cases(json.loads((PRIOR/'summary.json').read_text())['cases'])
  for (seed,arm),r in references.items():
   assert sha(r['generation']['artifacts']['video'])==r['video_sha256']
   if arm!='original':check_pair(references[(seed,'original')],r)
  instrument_hashes={str(p):sha(p) for p in FILES};reuse_summary=json.loads((REUSE/'summary.json').read_text());assert reuse_summary['exact_video_replay']
  assert reuse_summary['video_sha256']==references[(0,'original')]['video_sha256']
  reuse_protocol=json.loads((REUSE/'protocol.json').read_text())
  for path,digest in reuse_protocol['vae_hashes'].items():assert sha(path)==digest
  for name in ('dd2_capture_hooks.py','dd2_capture_python'):
   path=str(ROOT/'scripts'/name);assert instrument_hashes[path]==reuse_protocol['instrument_sha256'][path]
  reuse_hook=validate_hooks(Path(reuse_summary['capture_dir']))
  assert sha(reuse_summary['video'])==reuse_summary['video_sha256']
  assert sha(ROOT/'ENV.py')=='bb5d9a67c2cd2ca61946e77797dd9f5f219ed27b61ba98ecfb95ddf64c318e72'
  anchor={'prior_protocol_sha256':sha(PRIOR/'protocol.json'),'prior_summary_sha256':sha(PRIOR/'summary.json'),'reuse_summary_sha256':sha(REUSE/'summary.json'),'reuse_hook_sha256':sha(Path(reuse_summary['capture_dir'])/'hook_state.json'),'instrument_sha256':instrument_hashes}
  if not args.check_only:
   gate=json.loads((PREFLIGHT/'preflight.json').read_text());assert gate['passed'] and not gate['generation_run'];assert gate['anchor']==anchor
  spec={'experiment':'six_case_matched_lossless_capture_v1','written_before_new_generation':True,'check_only':args.check_only,'candidate_index':5596,'arms':list(ARMS),'seed_offsets':[0,1],
   'new_render_count':0 if args.check_only else 5,'reused_case':{'seed_offset':0,'arm':'original','run':str(REUSE)},'total_cases':6,'anchor':anchor,
   'request':asdict(REQUEST),'generation_settings':{'steps':50,'image_condition_scale':1,'seed_bank':0,'decode_chunk_size':1,'precision':'original fp16; framework defaults unchanged'},
   'old_cases':[{'seed_offset':seed,'arm':arm,'video':r['generation']['artifacts']['video'],'video_sha256':r['video_sha256']} for (seed,arm),r in references.items()],
   'capture':'Same proven hooks for all cases; final latent, float decoded tensor, all8 panorama/front PNG, all8 composite PNG, original MP4. Reused case copied into new output with provenance; no symlink to external data.',
   'analysis_plan':{'detector':'same FasterRCNN ResNet50 FPN V2 COCO_V1 checkpoint as prior analysis','confidence':.5,'unique_bus_iou':.3,'frame_count':48,
    'association':'Use each arm own registered projected box; exactly one candidate, no highest-score fallback.',
    'paired_delta':'Edited minus same-seed original on paired measurable frames; retain missing/ambiguous frames. Later-frame mean only if all frames4,5,6,7 measurable in BOTH videos.',
    'export_comparison':'Compare lossless and original MP4 for every frame, case and paired contrast. Never replace original scores or select favorable setting.',
    'human_annotations':'Existing annotations refer to prior MP4 images, not automatically ground truth for lossless PNG. Independently verify localization before interpreting small displacement.'},
   'failure_policy':'On mismatch preserve current evidence, stop batch, no automatic seed/parameter retry. All six cases required for complete result.',
   'scope':'Post-hoc export sensitivity audit of previously observed scene, not independent confirmatory evidence or six independent scenes. No paper score edits or efficacy claim.'}
  save(out/'protocol.json',spec)
  if args.check_only:
   checks=[]
   for arm in ARMS:
    d=out/f'audit_{arm}';d.mkdir();os.environ['DRIVELOOP_CAPTURE_DIR']=str(d/'capture');os.environ['DRIVELOOP_DD2_IMAGE_CONDITION_AUDIT_PATH']=str(d/'latent_audit.json')
    backend=make_backend(case['inputs'][arm],d/'artifacts',audit_only=True);backend.python_executable=str(ROOT/'scripts/dd2_capture_python')
    g=backend.generate(REQUEST,iteration=0);assert_native(g,case['inputs'][arm],case['runtime_audits'][arm]);save(d/'generation.json',asdict(g))
    h=json.loads((d/'capture/hook_state.json').read_text());assert set(h['installed'])=={PIPE,TESTER} and h['counts']=={'decode':0,'postprocess':0,'composite':0}
    assert 'video' not in g.artifacts and g.metadata['dd2_audit_only'] and not (d/'latent_audit.json').exists()
    checks.append({'arm':arm,'input_hashes_match':True,'hook_calls':h['counts']});print('INPUT_AUDIT_PASS',arm,'no inference',flush=True)
   check_fixed(pre);assert {str(p):sha(p) for p in FILES}==instrument_hashes
   save(out/'preflight.json',{'passed':True,'generation_run':False,'anchor':anchor,'all_3_condition_loaders_verified':checks,'reuse_capture_verified':True,'six_reference_videos_and_pairing_verified':True})
   print('MATCHED_CAPTURE_PREFLIGHT_PASS: three loader audits, six references, no generation.',flush=True);return
  rows=[]
  for seed in (0,1):
   for arm in ARMS:
    reference=references[(seed,arm)];d=out/f'candidate5596_seed{seed}_{arm}';t0=time.monotonic()
    check_fixed(pre);assert {str(p):sha(p) for p in FILES}==instrument_hashes
    if (seed,arm)==(0,'original'):
     source=REUSE/'candidate5596_seed0_original';shutil.copytree(source,d)
     generation=json.loads((d/'generation.json').read_text());original_generation=copy.deepcopy(generation)
     # Only relocate saved artifact pointers; model audit values stay unchanged.
     for key,path in generation['artifacts'].items():
      relative=Path(path).relative_to(source);generation['artifacts'][key]=str(d/relative);assert Path(generation['artifacts'][key]).exists()
     save(d/'generation.json',generation);save(d/'reuse_provenance.json',{'source_case':str(source),'source_summary_sha256':sha(REUSE/'summary.json'),'original_generation_record':original_generation})
    else:
     d.mkdir();os.environ['DRIVELOOP_CAPTURE_DIR']=str(d/'capture');os.environ['DRIVELOOP_DD2_IMAGE_CONDITION_AUDIT_PATH']=str(d/'latent_audit.json')
     backend=make_backend(case['inputs'][arm],d/'artifacts');backend.python_executable=str(ROOT/'scripts/dd2_capture_python')
     g=backend.generate(REQUEST,iteration=seed);assert_native(g,case['inputs'][arm],case['runtime_audits'][arm]);generation=asdict(g);save(d/'generation.json',generation)
    h=validate_hooks(d/'capture');video=Path(generation['artifacts']['video']);video_sha=sha(video);comparison=compare_videos(reference['generation']['artifacts']['video'],video)
    save(d/'video_comparison.json',comparison)
    row={'candidate_index':5596,'scene_token':case['scene_token'],'seed_offset':seed,'arm':arm,'generation':generation,
     'latent_audit':json.loads((d/'latent_audit.json').read_text()),'video_sha256':video_sha,'exact_video_replay':video_sha==reference['video_sha256'] and comparison['all_decoded_pixels_equal'],
     'video_comparison':comparison,'capture_dir':str(d/'capture'),'reused':(seed,arm)==(0,'original'),'seconds':time.monotonic()-t0}
    save(d/'result.json',row);rows.append(row);save(out/'partial_results.json',rows)
    validate_row(row,reference,case['runtime_audits'][arm])
    if arm!='original':check_pair(next(x for x in rows if x['seed_offset']==seed and x['arm']=='original'),row)
    print('REUSED_EXACT' if row['reused'] else 'GENERATED_EXACT',arm,'seed',seed,'all capture/replay checks passed',flush=True)
  final=index_cases(rows)
  for seed in (0,1):
   for arm in ('delayed','advanced'):check_pair(final[(seed,'original')],final[(seed,arm)])
  assert final[(0,'original')]['latent_audit']['initial_noise']['sha256']!=final[(1,'original')]['latent_audit']['initial_noise']['sha256']
  check_fixed(pre);assert {str(p):sha(p) for p in FILES}==instrument_hashes
  for r in references.values():assert sha(r['generation']['artifacts']['video'])==r['video_sha256']
  for path,digest in reuse_protocol['vae_hashes'].items():assert sha(path)==digest
  save(out/'summary.json',{'complete':True,'cases':rows,'total_cases':6,'new_render_count':5,'reused_count':1,'all_six_exact_video_replays':True,'paired_input_checks_passed':True,'all_original_videos_unchanged':True,'protocol':str(out/'protocol.json')})
  print('MATCHED_LOSSLESS_CAPTURE_COMPLETE',out,flush=True);print('ALL_SIX_EXACT_REPLAY_PASS',flush=True)
 except Exception as exc:
  save(out/'failure.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'evidence_retained':True,'no_automatic_retry':True});raise
if __name__=='__main__':main()
