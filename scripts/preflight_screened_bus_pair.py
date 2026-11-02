"""Replay actual source override and DD2 transform on CPU, before any render."""
import ast,copy,json,os,warnings
from pathlib import Path
import numpy as np,torch,cv2,mmengine
from PIL import Image,ImageDraw
from torchvision import transforms
from dreamer_datasets import load_dataset,boxes3d_utils
from driveloop.dd2_override import apply_dd2_override_to_sample,tensor_signature,write_override_audit
from scripts.run_screened_bus_pair import context,setup_env,make_backend,SOURCE,ROOT
from scripts.run_bus_height_pair import sha,save
setup_env();torch.set_num_threads(2);manifest,entries,m,request=context();dataset=load_dataset(manifest['dataset'])
args=(m['dd2_prompt'],m['dd2_structural_input_plan'],m['dd2_override_candidate_plan'],manifest['binding'])
on=make_backend(manifest,entries,SOURCE/'unused_preflight',True)._build_override_json(*args);off=make_backend(manifest,entries,SOURCE/'unused_preflight',False)._build_override_json(*args)
a=copy.deepcopy(on);b=copy.deepcopy(off);a['boxes3d']['per_frame_append_ego']=[];a['audit']['experimental_executed_plan']['actor_injection']=False;assert a==b
path=ROOT/'dreamer-train/projects/DriveDreamer2/drivedreamer2/drivedreamer2_transforms.py';tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DriveDreamer2_Transform');ns=dict(globals(),HEIGHT=900,WIDTH=1600);exec(compile(ast.Module(body=[cls],type_ignores=[]),str(path),'exec'),ns)
transform=ns['DriveDreamer2_Transform'](resolution=(256,448),gd_input_name='image_hdmap',bd_input_name='image_box',is_train=False,embed_map_path='/data/projects/DriveLoop/clip_text_transform_after_pool_panoramic.pkl',text_embed_dim=1024,box_normal=True)
def checked_transform(sample):
 with warnings.catch_warnings(record=True) as seen:
  warnings.simplefilter('always',RuntimeWarning);result=transform(sample)
 return result,[str(w.message) for w in seen]
rows=[];sheet=Image.new('RGB',(448*4,290*2),'white');draw=ImageDraw.Draw(sheet)
for i,e in enumerate(entries):
 for cam,idx in e['source_record_indices'].items():
  raw=dataset[idx];assert raw['scene_token']==manifest['scene_token'] and raw['cam_type']==cam
  support,_=apply_dd2_override_to_sample(copy.deepcopy(raw),off);actor,audit=apply_dd2_override_to_sample(copy.deepcopy(raw),on)
  assert len(support['boxes3d'])==len(raw['boxes3d'])
  before,before_warnings=checked_transform(support);after,after_warnings=checked_transform(actor)
  aa=before['box_downsampler_input'].numpy();bb=after['box_downsampler_input'].numpy();channels=[j for j in range(len(aa)) if not np.array_equal(aa[j],bb[j])]
  assert set(channels).issubset({11}),channels
  assert np.array_equal(before['grounding_downsampler_input'].numpy(),after['grounding_downsampler_input'].numpy())
  row={'support_raster_warnings':before_warnings,'actor_raster_warnings':after_warnings,'step':i,'camera':cam,'record_index':idx,'added_boxes':len(actor['boxes3d'])-len(raw['boxes3d']),'changed_channels':channels,'changed_bus_pixels':int(np.count_nonzero(aa[11]!=bb[11]))};rows.append(row)
  if cam=='cam_front':
   assert row['added_boxes']==1 and channels==[11] and row['changed_bus_pixels']>0,row
   im=raw['image'].convert('RGB').resize((448,252));d=ImageDraw.Draw(im);pts=boxes3d_utils.corners3d_to_corners2d(boxes3d_utils.boxes3d_to_corners3d(actor['boxes3d'][-1:],rot_axis=1),cam_intrinsic=raw['calib']['cam_intrinsic'])[0]*[448/1600,252/900]
   for u,v in transform.box_skeleton:d.line([tuple(pts[u]),tuple(pts[v])],fill='red',width=2)
   x=(i%4)*448;y=(i//4)*290;sheet.paste(im,(x,y+30));draw.text((x+4,y+5),f'candidate 1318 | frame {i} | RED planned bus',fill='black')
report={'passed':True,'gpu_generation':False,'entries_sha256':sha(SOURCE/'selected_entries.json'),'labels_sha256':sha(Path(manifest['dataset'])/'labels/data.pkl'),'transform_sha256':sha(path),'rows':rows,'checks':['actor/support overrides differ only by actor entries and explicit audit flag','source box count retained','only bus conditioning channel changes','map input identical','bus projection changes all eight front frames'],'claim':'Input transformation only, not video success or dynamics.'}
save(SOURCE/'runtime_preflight.json',report);sheet.save(SOURCE/'candidate1318_plan_projection.jpg',quality=95);print('RUNTIME_PREFLIGHT_PASS',len(rows),'camera frames',flush=True)
