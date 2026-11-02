import json,pickle
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from dreamer_datasets import load_dataset,boxes3d_utils
ROOT=Path('/home/lesliton/projects/DriveLoop');O=ROOT/'outputs/revision_20261008_signal_precheck';S=ROOT/'outputs/revision_20261007_native_multiscene_source/candidate1583';m=json.loads((S/'original/manifest.json').read_text());ds=load_dataset(m['dataset']);records=pickle.loads((Path(m['dataset'])/'labels/data.pkl').read_bytes());tok=m['instance_token']
mods={a:json.loads((O/'geometry'/a/'manifest.json').read_text()) for a in ('delayed025','advanced025')}
sheet=Image.new('RGB',(448*2,288*4),'white');d=ImageDraw.Draw(sheet)
for i in range(8):
 r=records[24+i*3];im=ds[24+i*3]['image'].convert('RGB').resize((448,256),Image.Resampling.BILINEAR);draw=ImageDraw.Draw(im)
 boxes={'original':r['boxes3d'][r['instance_tokens'].index(tok)],**{a:np.array(v['geometry']['frames'][i]['edited_box']) for a,v in mods.items()}}
 for a,b in boxes.items():
  cs=boxes3d_utils.boxes3d_to_corners3d(np.array(b)[None],rot_axis=1);pts=boxes3d_utils.corners3d_to_corners2d(cs,cam_intrinsic=r['calib']['cam_intrinsic'])[0]*np.array([448/1600,256/900]);box=[*pts.min(0),*pts.max(0)]
  draw.rectangle(box,outline={'original':'white','delayed025':'cyan','advanced025':'yellow'}[a],width=2)
 x=i%2*448;y=i//2*288;sheet.paste(im,(x,y+32));d.text((x+4,y+7),f'Frame {i} | original:white delay:cyan advance:yellow',fill='black')
sheet.save(O/'candidate1583_source_plan.jpg',quality=95)
q={'candidate_index':1583,'scene_token':m['scene_token'],'instance_token':tok,'scene_description':m['scene_description'],'source_dataset':m['dataset'],'source_only':True,'geometry_summary':{}}
for a,v in mods.items():
 fs=v['geometry']['frames'];ints=v['geometry']['intervals'];q['geometry_summary'][a]={'late_mean_dx':float(np.mean([f['projected_center_shift_x_px_at448'] for f in fs[4:]])),'minimum_sampled_clearance_m':min(f['clearance_m'] for f in fs),'minimum_interval_clearance_m':min(f['clearance_m'] for f in ints),'minimum_road_fraction':min(f['road_fraction'] for f in fs),'minimum_visible_fraction':min(f['visible_fraction'] for f in fs),'maximum_absolute_bottom_change_m':max(abs(f['bottom_change_m']) for f in fs),'changed_records':len(v['changed_records']),'native_labels_sha256':v['native_labels_sha256'],'edited_labels_sha256':v['edited_labels_sha256']}
(O/'candidate_summary.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q),flush=True)
