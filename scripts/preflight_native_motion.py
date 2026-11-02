"""Verify pose-only modifications and actual raster channels, then audit real loading."""
import ast,copy,json,pickle,warnings,os
from pathlib import Path
import numpy as np,torch,cv2,mmengine
from torchvision import transforms
from dreamer_datasets import load_dataset,boxes3d_utils
from driveloop.dd2_override import apply_dd2_override_to_sample,tensor_signature,write_override_audit
from scripts.run_native_motion_pair import ROOT,SOURCE,BASELINE,context
from scripts.run_native_bus_control import setup_env,make_backend,REQUEST,assert_native
from scripts.run_bus_height_pair import sha,save
from dataclasses import asdict
setup_env();torch.set_num_threads(2);m,edited,baseline,bp=context();raw=load_dataset(m['native_dataset']);changed_records=pickle.loads((Path(m['edited_dataset'])/'labels/data.pkl').read_bytes())
path=ROOT/'dreamer-train/projects/DriveDreamer2/drivedreamer2/drivedreamer2_transforms.py'
cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='DriveDreamer2_Transform');ns=dict(globals(),HEIGHT=900,WIDTH=1600)
exec(compile(ast.Module(body=[cls],type_ignores=[]),str(path),'exec'),ns)
t=ns['DriveDreamer2_Transform'](resolution=(256,448),gd_input_name='image_hdmap',bd_input_name='image_box',is_train=False,
 embed_map_path='/data/projects/DriveLoop/clip_text_transform_after_pool_panoramic.pkl',text_embed_dim=1024,box_normal=True)
for part in ('images','hdmaps'):
 assert (Path(m['edited_dataset'])/part).resolve()==(Path(m['native_dataset'])/part).resolve()
rows=[]
for cam in range(6):
 for step in range(8):
  idx=cam*24+step*3;a=raw[idx];b=copy.deepcopy(a);b['boxes3d']=changed_records[idx]['boxes3d'].copy()
  assert np.array_equal(a['image'],b['image']) and np.array_equal(a['image_hdmap'],b['image_hdmap'])
  assert a['ori_labels3d']==b['ori_labels3d'] and a['instance_tokens']==b['instance_tokens']
  assert len(a['boxes3d'])==len(b['boxes3d'])
  diff=np.any(a['boxes3d']!=b['boxes3d'],axis=1)
  for j in np.flatnonzero(diff):assert a['instance_tokens'][j]==m['instance_token']
  with warnings.catch_warnings():
   warnings.simplefilter('error',RuntimeWarning);aa=t(a);bb=t(b)
  x=aa['box_downsampler_input'].numpy();y=bb['box_downsampler_input'].numpy();channels=[c for c in range(19) if not np.array_equal(x[c],y[c])]
  assert channels==([11] if cam==1 and step>0 else []),(cam,step,channels)
  assert np.array_equal(aa['grounding_downsampler_input'],bb['grounding_downsampler_input'])
  assert np.array_equal(aa['input_image'],bb['input_image']) and np.array_equal(aa['prompt_embeds'],bb['prompt_embeds'])
  rows.append({'camera':cam,'step':step,'changed_channels':channels,'changed_pixels':int(np.count_nonzero(x!=y))})
print('POSE_PREFLIGHT_PASS: first frame/all source images unchanged; only 7 front bus-channel tensors differ.',flush=True)
g=make_backend(edited,SOURCE/'input_audit',audit_only=True).generate(REQUEST,iteration=0);assert_native(g,edited)
a=g.metadata['dd2_runtime_input_audit'];old=baseline['cases'][0]['generation']['metadata']['dd2_runtime_input_audit']
for key in ('img_cond','grounding_downsampler_input','prompt_embed'):assert a[key]['sha256']==old[key]['sha256'],key
assert a['box_downsampler_input']['sha256']!=old['box_downsampler_input']['sha256'];assert 'video' not in g.artifacts
save(SOURCE/'audit_generation.json',asdict(g))
files=[ROOT/'scripts/run_native_motion_pair.py',ROOT/'scripts/preflight_native_motion.py',ROOT/'scripts/prepare_native_motion.py',
 SOURCE/'manifest.json',SOURCE/'geometry_checks.json',SOURCE/'selection_protocol.json',Path(m['edited_dataset'])/'labels/data.pkl']
files.extend((Path(m['edited_dataset'])/'labels').glob('config.json'));files.append(Path(m['edited_dataset'])/'config.json')
save(SOURCE/'preflight.json',{'passed':True,'generation_run':False,'rows':rows,'runtime_input_audit':a,'fixed_hashes':{str(p):sha(p) for p in files},
 'claim':'Only existing target pose changes; image/map/text preserved. Not generated-video success.'})
print('MOTION_RUNTIME_PREFLIGHT_PASS: generation skipped.',flush=True)
