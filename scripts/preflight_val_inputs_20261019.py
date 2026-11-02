"""Source-byte, pose-only, raster-channel and actual-loader audits; no diffusion."""
import ast,copy,json,pickle,warnings,os,hashlib
from pathlib import Path
from dataclasses import asdict
import numpy as np,torch,cv2,mmengine,lmdb
from torchvision import transforms
from dreamer_datasets import load_dataset,boxes3d_utils
from driveloop.dd2_override import apply_dd2_override_to_sample,tensor_signature,write_override_audit
from scripts.run_native_bus_control import ROOT,setup_env,make_backend,REQUEST,assert_native
from scripts.run_bus_height_pair import sha,save
SOURCE=None

def main():
 global SOURCE
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--candidate',type=int,choices=[568,629],required=True);args=parser.parse_args()
 SOURCE=ROOT/'outputs/revision_20261019_val_inputs'/f'val_candidate{args.candidate}'
 assert not (SOURCE/'preflight.json').exists(), 'Do not overwrite completed preflight'
 setup_env();torch.set_num_threads(2);manifest=json.loads((SOURCE/'manifest.json').read_text());assert manifest['selected_count']==1 and manifest['selected'][0]['source']['candidate_index']==args.candidate
 from driveloop.native_phase import digest
 initial_hashes=json.loads((SOURCE/'launch_gate.json').read_text())['fixed_hashes']
 for file,h in initial_hashes.items():assert digest(file)==h,file
 path=ROOT/'dreamer-train/projects/DriveDreamer2/drivedreamer2/drivedreamer2_transforms.py'
 cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='DriveDreamer2_Transform');ns=dict(globals(),HEIGHT=900,WIDTH=1600)
 exec(compile(ast.Module(body=[cls],type_ignores=[]),str(path),'exec'),ns)
 t=ns['DriveDreamer2_Transform'](resolution=(256,448),gd_input_name='image_hdmap',bd_input_name='image_box',is_train=False,embed_map_path='/data/projects/DriveLoop/clip_text_transform_after_pool_panoramic.pkl',text_embed_dim=1024,box_normal=True)
 cases=[];files=[SOURCE/'manifest.json',SOURCE/'protocol.json',SOURCE/'selection_ledger.json',ROOT/'ENV.py',Path('/data/projects/DriveLoop/clip_text_transform_after_pool_panoramic.pkl')]
 for entry in manifest['selected']:
  m=entry['source'];case=Path(entry['case']);native=Path(m['dataset']);records=pickle.loads((native/'labels/data.pkl').read_bytes());selected=pickle.loads((case/'original/selected_window.pkl').read_bytes());original=[r for group in selected['groups'] for r in group]
  assert len(records)==len(original)==144
  for a,b in zip(original,records):
   a=copy.deepcopy(a);b=copy.deepcopy(b)
   for key in ('data_index','multiview_start_idx'):a.pop(key,None);b.pop(key,None)
   assert pickle.dumps(a,protocol=4)==pickle.dumps(b,protocol=4)
  for part in ('images','hdmaps'):
   env=lmdb.open(str(native/part),readonly=True,lock=False,readahead=False)
   with env.begin() as tx:
    for i,h in enumerate(m['image_and_map_hashes'][part]):assert hashlib.sha256(tx.get(str(i).encode())).hexdigest()==h
   env.close()
  raw=load_dataset(str(native));raster=[];audits={};inputs={}
  for arm in ('original','delayed','advanced'):
   gm=entry['arms'][arm];assert gm['geometry']['passed'];inputs[arm]=copy.deepcopy(m)
   if arm!='original':inputs[arm]['dataset']=gm['edited_dataset']
   edited=pickle.loads((Path(inputs[arm]['dataset'])/'labels/data.pkl').read_bytes());changes=0
   for part in ('images','hdmaps'):assert (Path(inputs[arm]['dataset'])/part).resolve()==(native/part).resolve()
   for idx,(a,b) in enumerate(zip(records,edited)):
    assert a.keys()==b.keys()
    # Independent pickle loads can preserve shared references differently; compare each field, not object-graph memo bytes.
    for key in a:
     if key!='boxes3d':assert pickle.dumps(a[key],protocol=4)==pickle.dumps(b[key],protocol=4),(arm,idx,key)
    changed=np.flatnonzero(np.any(a['boxes3d']!=b['boxes3d'],axis=1))
    for j in changed:
     assert a['instance_tokens'][j]==m['instance_token'] and idx%24 in range(3,22,3)
     assert np.array_equal(a['boxes3d'][j,3:7],b['boxes3d'][j,3:7]) and a['boxes3d'][j,8]==b['boxes3d'][j,8]
   for cam in range(6):
    for step in range(8):
     idx=cam*24+step*3;a=raw[idx];b=copy.deepcopy(a);b['boxes3d']=edited[idx]['boxes3d'].copy()
     with warnings.catch_warnings():
      warnings.simplefilter('error',RuntimeWarning);aa=t(a);bb=t(b)
     x=aa['box_downsampler_input'].numpy();y=bb['box_downsampler_input'].numpy();channels=[c for c in range(19) if not np.array_equal(x[c],y[c])]
     assert set(channels)<={11},(arm,cam,step,channels)
     if step==0 or arm=='original':assert not channels
     for k in ('grounding_downsampler_input','input_image','prompt_embeds'):assert np.array_equal(aa[k],bb[k])
     changes+=bool(channels);raster.append({'arm':arm,'camera':cam,'step':step,'changed_channels':channels,'changed_pixels':int(np.count_nonzero(x!=y))})
   assert arm=='original' or changes>0
   os.environ['DRIVELOOP_CAPTURE_DIR']=str(case/f'audit_{arm}'/'capture')
   backend=make_backend(inputs[arm],case/f'audit_{arm}',audit_only=True);backend.python_executable=str(ROOT/'scripts/dd2_capture_python')
   g=backend.generate(REQUEST,iteration=0);h=json.loads((case/f'audit_{arm}'/'capture/hook_state.json').read_text())
   from scripts.dd2_capture_hooks import PIPE,TESTER
   assert set(h['installed'])=={PIPE,TESTER} and h['counts']=={'decode':0,'postprocess':0,'composite':0}
   assert_native(g,inputs[arm]);assert 'video' not in g.artifacts and g.metadata['dd2_audit_only']
   audits[arm]=g.metadata['dd2_runtime_input_audit'];save(case/f'audit_{arm}/generation.json',asdict(g))
   if arm!='original':
    for k in ('img_cond','grounding_downsampler_input','prompt_embed'):assert audits[arm][k]['sha256']==audits['original'][k]['sha256']
    assert audits[arm]['box_downsampler_input']['sha256']!=audits['original']['box_downsampler_input']['sha256']
   print('AUDIT_PASS',m['candidate_index'],arm,'changed view-frame tensors',changes,flush=True)
  del raw
  cases.append({'candidate_index':m['candidate_index'],'scene_token':m['scene_token'],'inputs':inputs,'runtime_audits':audits,'raster_audit':raster})
  files.extend(case.rglob('*.json'));files.extend(case.rglob('*.pkl'))
  files.extend(native/part/'data.mdb' for part in ('images','hdmaps'))
  files.extend(native.rglob('config.json'));files.append(native/'labels/data.pkl');files.append(Path(m['identity']))
  for a in inputs.values():
   files.append(Path(a['dataset'])/'labels/data.pkl');files.extend(Path(a['dataset']).rglob('config.json'))
 files.extend(Path(p) for p in initial_hashes)
 for folder in ('driveloop','dreamer-train/projects/DriveDreamer2','dreamer-models/dreamer_models','dreamer-datasets/dreamer_datasets'):files.extend((ROOT/folder).rglob('*.py'))
 for name in ('prepare_native_multiscene','preflight_native_multiscene','run_native_multiscene','run_native_bus_control','run_native_advance_pair','run_bus_height_pair'):files.append(ROOT/'scripts'/f'{name}.py')
 files.append(ROOT/'scripts/run_native_multiscene.sh')
 for name in ('prepare_candidate1583_inputs.py','preflight_candidate1583.py','run_candidate1583_capture.py','run_candidate1583_capture.sh','dd2_capture_hooks.py','dd2_capture_python','run_native_matched_capture.py','run_native_generation_capture.py'):
  files.append(ROOT/'scripts'/name)
 reuse=json.loads((ROOT/'outputs/native_generation_capture_20261008_131957_51857/protocol.json').read_text())
 for p,h in reuse['vae_hashes'].items():assert sha(p)==h;files.append(Path(p))
 for name in ('dd2_capture_hooks.py','dd2_capture_python'):
  p=str(ROOT/'scripts'/name);assert sha(p)==reuse['instrument_sha256'][p]
 detector=Path('/data/models/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth');assert sha(detector)=='dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf';files.append(detector)
 old=json.loads((ROOT/'outputs/native_bus_control_20261006_193126_95686/protocol.json').read_text());weight=os.environ['DRIVELOOP_DD2_WEIGHT_PATH'];assert sha(weight)==old['weight_sha256']
 assert sha(ROOT/'ENV.py')=='bb5d9a67c2cd2ca61946e77797dd9f5f219ed27b61ba98ecfb95ddf64c318e72'
 for file,h in initial_hashes.items():assert digest(file)==h,file
 save(SOURCE/'preflight.json',{'passed':True,'generation_run':False,'cases':cases,'render_count':0,'actual_loader_calls':3,'diffusion_calls':0,'weight':weight,'weight_sha256':old['weight_sha256'],'fixed_hashes':{str(p):sha(p) for p in sorted(set(files))}})
 from driveloop.native_phase import NativePhaseRegistry
 r=NativePhaseRegistry(SOURCE/'preflight.json',digest(SOURCE/'preflight.json'))
 assert r.candidate==args.candidate
 save(SOURCE/'registry_check.json',{'passed':True,'preflight_sha256':digest(SOURCE/'preflight.json'),'candidate_index':r.candidate,'scene_token':r.scene,'instance_token':r.instance,'target_means':r.means,'generation_run':False,'actual_loader_calls':3,'diffusion_calls':0})
 print('VAL_INPUT_PREFLIGHT_COMPLETE',r.means,'3 loaders; no diffusion',flush=True)
if __name__=='__main__':main()
