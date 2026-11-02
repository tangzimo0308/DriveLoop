"""Replay exactly one preregistered native-source case, with lossless observations."""
import argparse,json,os,sys,time,traceback
from dataclasses import asdict
from pathlib import Path
from scripts.run_native_bus_control import ROOT,setup_env,make_backend,REQUEST,assert_native
from scripts.dd2_capture_hooks import PIPE,TESTER,sha,save
PRIOR=ROOT/'outputs/native_multiscene_20261007_214304_13198'
PREFLIGHT=ROOT/'outputs/revision_20261008_generation_capture_preflight'
INSTRUMENTS=[ROOT/'scripts'/n for n in ('dd2_capture_hooks.py','dd2_capture_python','run_native_generation_capture.py','run_native_generation_capture.sh')]

def check_fixed(pre):
 for path,digest in pre['fixed_hashes'].items():assert sha(path)==digest,('Frozen input/code changed',path)
 assert sha(pre['weight'])==pre['weight_sha256'],'DD2 weight changed'

def compare_videos(old,new):
 import cv2,numpy as np
 a=cv2.VideoCapture(str(old));b=cv2.VideoCapture(str(new));records=[]
 try:
  for i in range(8):
   ok,x=a.read();ok2,y=b.read();assert ok and ok2,('Missing frame',i)
   assert x.shape==y.shape==(784,2688,3)
   delta=np.abs(x.astype(np.int16)-y.astype(np.int16))
   records.append({'frame':i,'equal':bool(np.array_equal(x,y)),'differing_values':int(np.count_nonzero(delta)),'max_abs_difference':int(delta.max()),'mean_abs_difference':float(delta.mean())})
  assert a.read()[0] is False and b.read()[0] is False,'Unexpected extra frames'
 finally:a.release();b.release()
 return {'all_decoded_pixels_equal':all(x['equal'] for x in records),'frames':records,'library':'cv2 '+cv2.__version__,'comparison':'All pixels of original 2688x784 saved composites; no resizing or threshold tolerance.'}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output-dir',required=True,type=Path);parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
 setup_env();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
 try:
  protocol=json.loads((PRIOR/'protocol.json').read_text());pre=protocol['preflight'];assert pre['passed'] and not pre['generation_run']
  case=next(c for c in pre['cases'] if c['candidate_index']==5596)
  rows=json.loads((PRIOR/'summary.json').read_text())['cases'];reference=next(r for r in rows if r['candidate_index']==5596 and r['arm']=='original' and r['seed_offset']==0)
  oldvideo=Path(reference['generation']['artifacts']['video'])
  assert sha(oldvideo)==reference['video_sha256'];check_fixed(pre)
  fixed_instruments={str(p):sha(p) for p in INSTRUMENTS}
  if not args.check_only:
   gate=json.loads((PREFLIGHT/'preflight.json').read_text());assert gate['passed'] and not gate['generation_run']
   assert gate['instrument_sha256']==fixed_instruments,'Instrumentation changed since preflight'
   assert gate['prior_protocol_sha256']==sha(PRIOR/'protocol.json')
   assert gate['reference_video_sha256']==reference['video_sha256']
  vae_root=Path('/data/projects/DriveLoop/hf_cache/stable-video-diffusion-img2vid-xt-1-1/vae')
  vae_hashes={str(vae_root/n):sha(vae_root/n) for n in ('config.json','diffusion_pytorch_model.fp16.safetensors')}
  prior_vae=json.loads((ROOT/'outputs/revision_20261008_vae_check/protocol.json').read_text())
  assert vae_hashes[str(vae_root/'diffusion_pytorch_model.fp16.safetensors')]==prior_vae['vae_weight_sha256']
  assert vae_hashes[str(vae_root/'config.json')]==prior_vae['vae_config_sha256']
  spec={'experiment':'native_generation_capture_v1','written_before_generation':True,'candidate_index':5596,'arm':'original','seed_offset':0,'render_count':0 if args.check_only else 1,
   'selection_rule':'Previously registered original condition, first registered seed (0); selected before new render, not by detector score.',
   'request':asdict(REQUEST),'steps':50,'image_condition_scale':1,'seed_bank':0,'decode_chunk_size':1,
   'same_environment':'Existing drivedreamer environment; preserve framework precision/TF32 defaults. No new environment, model, detector or threshold.',
   'prior_protocol':str(PRIOR/'protocol.json'),'prior_protocol_sha256':sha(PRIOR/'protocol.json'),'prior_summary_sha256':sha(PRIOR/'summary.json'),
   'reference_video':str(oldvideo),'reference_video_sha256':reference['video_sha256'],'reference_latent_audit':reference['latent_audit'],
   'reference_runtime_audit':case['runtime_audits']['original'],'instrument_sha256':fixed_instruments,'vae_hashes':vae_hashes,
   'capture':'Final post-diffusion scaled latent at decode entry; float32 decoded tensor before display; all8 RGB panorama and front PNGs after actual tensor2vid; all8 full composite PNGs before unchanged MP4 writer; original MP4.',
   'noninterference':'In-process import hooks call original functions once and return original objects. Save copied/detached data; audit observer RNG and generated/composite equality. No model source edits.',
   'verification':'Exact frozen files, weights, runtime input signatures, initial noise and conditioning latent, MP4 SHA, and all decoded MP4 pixels. No tolerance-based relabeling of mismatch.',
   'failure_policy':'Keep every result; a mismatch is marked and not accepted as an exact replay. No automatic seed retry or parameter changes.',
   'limits':'One native-bus diagnostic, not new actor/cut-in success or paper efficacy. Old run did not save final latent/pre-MP4 PNG, so equality at these stages cannot be verified retrospectively.',
   'check_only':args.check_only}
  save(out/'protocol.json',spec)
  d=out/'candidate5596_seed0_original';d.mkdir();os.environ['DRIVELOOP_CAPTURE_DIR']=str(d/'capture');os.environ['DRIVELOOP_DD2_IMAGE_CONDITION_AUDIT_PATH']=str(d/'latent_audit.json')
  backend=make_backend(case['inputs']['original'],d/'artifacts',audit_only=args.check_only)
  backend.python_executable=str(ROOT/'scripts/dd2_capture_python')
  t0=time.monotonic();g=backend.generate(REQUEST,iteration=0);save(d/'generation.json',asdict(g));assert_native(g,case['inputs']['original'],case['runtime_audits']['original'])
  hooks=json.loads((d/'capture/hook_state.json').read_text());assert set(hooks['installed'])=={PIPE,TESTER} and len(hooks['installed'])==2
  if args.check_only:
   assert hooks['counts']=={'decode':0,'postprocess':0,'composite':0};assert 'video' not in g.artifacts and g.metadata['dd2_audit_only']
   assert not (d/'latent_audit.json').exists()
   check_fixed(pre);assert {str(p):sha(p) for p in INSTRUMENTS}==fixed_instruments
   result={'passed':True,'generation_run':False,'capture_hooks_imported':True,'inference_hook_calls':hooks['counts'],'actual_loader_inputs_match_prior':True,'instrument_sha256':fixed_instruments,
    'prior_protocol_sha256':spec['prior_protocol_sha256'],'reference_video_sha256':reference['video_sha256'],'seconds':time.monotonic()-t0}
   save(out/'preflight.json',result);print('CAPTURE_PREFLIGHT_PASS: actual loader and both hooks verified; no inference.',flush=True);return
  assert hooks['counts']=={'decode':1,'postprocess':1,'composite':1}
  assert hooks['capture_rng_unchanged']==['before_decode','after_decode','after_postprocess','after_composite']
  assert hooks['generated_equals_composite_bottom']
  for name,h in hooks['files'].items():assert sha(d/'capture'/name)==h,name
  latent=json.loads((d/'latent_audit.json').read_text());save(out/'latent_comparison.json',{'current':latent,'reference':reference['latent_audit'],'equal':latent==reference['latent_audit']})
  assert latent==reference['latent_audit'],'Initial noise or image-conditioning latent differs'
  video=Path(g.artifacts['video']);video_sha=sha(video);comparison=compare_videos(oldvideo,video);save(out/'video_comparison.json',comparison)
  check_fixed(pre);assert sha(oldvideo)==reference['video_sha256'];assert all(sha(p)==h for p,h in vae_hashes.items());assert {str(p):sha(p) for p in INSTRUMENTS}==fixed_instruments
  exact=video_sha==reference['video_sha256'] and comparison['all_decoded_pixels_equal']
  result={'capture_complete':True,'exact_video_replay':exact,'video_sha256_match':video_sha==reference['video_sha256'],'all_decoded_pixels_equal':comparison['all_decoded_pixels_equal'],
   'input_and_initial_latent_checks_passed':True,'capture_observers_rng_unchanged':True,'generated_pixels_equal_composite_bottom':True,'reference_video_unchanged':True,
   'video':str(video),'video_sha256':video_sha,'reference_video_sha256':reference['video_sha256'],'capture_dir':str(d/'capture'),'seconds':time.monotonic()-t0,
   'interpretation':'Exact saved-video replay; captured intermediates support further direct checks.' if exact else 'REPLAY_MISMATCH: retain evidence; do not interpret captured intermediates as exact original-run intermediates.'}
  save(out/'summary.json',result)
  print('GENERATION_CAPTURE_COMPLETE',out,flush=True)
  print('EXACT_REPLAY_PASS' if exact else 'REPLAY_MISMATCH_REQUIRES_REVIEW',flush=True)
  if not exact:raise SystemExit(2)
 except Exception as exc:
  save(out/'failure.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'all_existing_artifacts_retained':True})
  raise
if __name__=='__main__':main()
