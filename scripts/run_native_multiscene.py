"""User-run matched original/delay/advance renders, two seeds per admitted scene."""
import argparse,json,os,time
from pathlib import Path
from dataclasses import asdict
from scripts.run_native_bus_control import ROOT,setup_env,make_backend,REQUEST,assert_native
from scripts.run_native_advance_pair import check_pair
from scripts.run_bus_height_pair import sha,save
SOURCE=ROOT/'outputs/revision_20261007_native_multiscene_source'

def check_files(pre):
 for p,h in pre['fixed_hashes'].items():assert sha(p)==h,p
 assert sha(pre['weight'])==pre['weight_sha256']

def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--check-only',action='store_true');a=p.parse_args()
 setup_env();pre=json.loads((SOURCE/'preflight.json').read_text());assert pre['passed'] and pre['generation_run'] is False and pre['render_count']>0
 check_files(pre);out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
 save(out/'protocol.json',{'preflight':pre,'preflight_sha256':sha(SOURCE/'preflight.json'),'selection':json.loads((SOURCE/'manifest.json').read_text()),'request':asdict(REQUEST),'check_only':a.check_only,'arms':['original','delayed','advanced'],'seed_offsets':[0,1],'claim':'Native-object editing diagnostic, no new-actor or closed-loop efficacy claim.'})
 print('MULTISCENE_PREFLIGHT_PASS',pre['render_count'],'renders',flush=True)
 if a.check_only:return
 rows=[]
 for case in pre['cases']:
  for seed in (0,1):
   baseline=None
   for arm in ('original','delayed','advanced'):
    check_files(pre);d=out/f"candidate{case['candidate_index']}_seed{seed}_{arm}";d.mkdir();os.environ['DRIVELOOP_DD2_IMAGE_CONDITION_AUDIT_PATH']=str(d/'latent_audit.json')
    start=time.monotonic();g=make_backend(case['inputs'][arm],d/'artifacts').generate(REQUEST,iteration=seed);assert_native(g,case['inputs'][arm],case['runtime_audits'][arm]);latent=json.loads((d/'latent_audit.json').read_text());assert latent['scale']==1 and latent['image_latent_before']==latent['image_latent_after']
    row={'candidate_index':case['candidate_index'],'scene_token':case['scene_token'],'seed_offset':seed,'arm':arm,'generation':asdict(g),'latent_audit':latent,'video_sha256':sha(g.artifacts['video']),'seconds':time.monotonic()-start}
    save(d/'generation.json',row);rows.append(row);save(out/'partial_results.json',rows)
    if arm=='original':baseline=row
    else:check_pair(baseline,row)
    print('GENERATED',case['candidate_index'],arm,'seed',seed,round(row['seconds'],2),'seconds; input checks passed',flush=True)
  originals=[r for r in rows if r['candidate_index']==case['candidate_index'] and r['arm']=='original'];assert len(originals)==2 and originals[0]['latent_audit']['initial_noise']['sha256']!=originals[1]['latent_audit']['initial_noise']['sha256']
 check_files(pre);assert len(rows)==pre['render_count']
 save(out/'summary.json',{'complete':True,'cases':rows,'paired_input_checks_passed':True,'render_count':len(rows),'protocol':str(out/'protocol.json')})
 print('EXPERIMENT_COMPLETE',out,flush=True)
if __name__=='__main__':main()
