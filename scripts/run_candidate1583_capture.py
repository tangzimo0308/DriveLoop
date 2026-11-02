"""Six new matched candidate1583 renders with lossless captures. User-run only."""
import argparse,json,os,time,traceback
from pathlib import Path
from dataclasses import asdict
from scripts.run_native_bus_control import ROOT,setup_env,make_backend,REQUEST,assert_native
from scripts.run_native_advance_pair import check_pair
from scripts.run_native_matched_capture import validate_hooks
from scripts.dd2_capture_hooks import sha,save
SOURCE=ROOT/'outputs/revision_20261008_candidate1583_inputs'
ARMS=('original','delayed','advanced');EXPECTED={(s,a) for s in (0,1) for a in ARMS}
def index_cases(rows):
 result={}
 for row in rows:
  assert row['candidate_index']==1583,'Unexpected candidate'
  key=(row['seed_offset'],row['arm']);assert key not in result,('Duplicate case',key);result[key]=row
 assert set(result)==EXPECTED,('Incomplete or extra cases',set(result))
 assert len({r['scene_token'] for r in rows})==1,'Mixed scenes'
 return result

def check_files(pre):
 for p,h in pre['fixed_hashes'].items():assert sha(p)==h,('Frozen input changed',p)
 assert sha(pre['weight'])==pre['weight_sha256'],'Model weight changed'

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
 setup_env();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
 try:
  pre=json.loads((SOURCE/'preflight.json').read_text());assert pre['passed'] and not pre['generation_run'] and pre['render_count']==6
  check_files(pre);assert len(pre['cases'])==1;case=pre['cases'][0];assert case['candidate_index']==1583
  protocol=json.loads((SOURCE/'protocol.json').read_text());assert protocol['seed_offsets']==[0,1] and protocol['arms']==list(ARMS)
  assert protocol['phase_seconds']=={'original':0,'delayed':.25,'advanced':-.25}
  protocol.update({'written_before_generation':True,'check_only':args.check_only,'preflight':pre,'preflight_sha256':sha(SOURCE/'preflight.json'),'selection':json.loads((SOURCE/'manifest.json').read_text()),'request':asdict(REQUEST),'new_render_count':0 if args.check_only else 6,'reuse_count':0})
  save(out/'protocol.json',protocol);print('CANDIDATE1583_FROZEN_INPUTS_PASS',flush=True)
  if args.check_only:
   save(out/'check_only.json',{'passed':True,'generation_run':False,'preflight_sha256':sha(SOURCE/'preflight.json')});return
  rows=[]
  for seed in (0,1):
   for arm in ARMS:
    check_files(pre);assert sha(SOURCE/'preflight.json')==protocol['preflight_sha256']
    dest=out/f'candidate1583_seed{seed}_{arm}';dest.mkdir();start=time.monotonic()
    os.environ['DRIVELOOP_CAPTURE_DIR']=str(dest/'capture');os.environ['DRIVELOOP_DD2_IMAGE_CONDITION_AUDIT_PATH']=str(dest/'latent_audit.json')
    backend=make_backend(case['inputs'][arm],dest/'artifacts');backend.python_executable=str(ROOT/'scripts/dd2_capture_python')
    g=backend.generate(REQUEST,iteration=seed);save(dest/'generation.json',asdict(g));assert_native(g,case['inputs'][arm],case['runtime_audits'][arm])
    latent=json.loads((dest/'latent_audit.json').read_text());assert latent['scale']==1 and latent['image_latent_before']==latent['image_latent_after']
    validate_hooks(dest/'capture')
    row={'candidate_index':1583,'scene_token':case['scene_token'],'seed_offset':seed,'arm':arm,'generation':asdict(g),'latent_audit':latent,'video_sha256':sha(g.artifacts['video']),'capture_dir':str(dest/'capture'),'seconds':time.monotonic()-start}
    save(dest/'result.json',row);rows.append(row);save(out/'partial_results.json',rows)
    if arm!='original':check_pair(next(r for r in rows if r['seed_offset']==seed and r['arm']=='original'),row)
    print('GENERATED_CAPTURED',arm,'seed',seed,round(row['seconds'],2),'seconds; input/noise/capture checks passed',flush=True)
  final=index_cases(rows)
  for seed in (0,1):
   for arm in ('delayed','advanced'):check_pair(final[(seed,'original')],final[(seed,arm)])
  assert final[(0,'original')]['latent_audit']['initial_noise']['sha256']!=final[(1,'original')]['latent_audit']['initial_noise']['sha256']
  for row in rows:validate_hooks(Path(row['capture_dir']));assert sha(row['generation']['artifacts']['video'])==row['video_sha256']
  check_files(pre);assert sha(SOURCE/'preflight.json')==protocol['preflight_sha256']
  save(out/'summary.json',{'complete':True,'cases':rows,'new_render_count':6,'total_cases':6,'paired_input_checks_passed':True,'all_captures_validated':True,'protocol':str(out/'protocol.json')})
  print('CANDIDATE1583_CAPTURE_COMPLETE',out,flush=True);print('ALL_SIX_INPUT_AND_CAPTURE_CHECKS_PASS',flush=True)
 except Exception as e:
  save(out/'failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(),'evidence_retained':True,'no_automatic_retry':True});raise
if __name__=='__main__':main()
