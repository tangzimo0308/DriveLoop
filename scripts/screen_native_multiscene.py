"""Select a native bus using source labels only, then copy its unchanged window."""
import copy,json,pickle,hashlib
from pathlib import Path
import cv2,numpy as np,lmdb
from dreamer_datasets import boxes3d_utils,load_dataset
from driveloop.source_sample_binding import build_source_sample_binding
from scripts.run_bus_height_pair import sha,save
from PIL import Image,ImageDraw
ROOT=Path('/home/lesliton/projects/DriveLoop');OUT=ROOT/'outputs/revision_20261007_multiscene_screen'
OUT.mkdir(exist_ok=False)
cache=ROOT/'outputs/revision_20261006_source_screen_expanded/shortlist.pkl'
shortlist=pickle.loads(cache.read_bytes());assert len(shortlist)==337
protocol={'shortlist':str(cache),'shortlist_sha256':sha(cache),'source_pool':337,'selection':'Visibility inventory in ascending candidate order; first visible continuous-24-frame rigid bus per window; excludes prior candidate91 scene; source only; no generation',
 'rules':{'category':'vehicle.bus.rigid','all_8_frames':True,'all_corners_depth_min_m':.5,'projected_visible_fraction_min':.95,'projected_convex_area_fraction_min':.025},
 'pool_limit':'Reuses prior 337-window shortlist, which excluded candidate28 scene and capped 2 windows per scene. Not an unbiased dataset sample.',
 'claim':'Source visibility positive-control selection; not method evaluation or new maneuver generation.'}
save(OUT/'selection_protocol.json',protocol)
ledger=[];selected=None;admitted=[];visible_image=np.array([[0,0],[1600,0],[1600,900],[0,900]],np.float32)
for c in sorted(shortlist,key=lambda x:x['candidate_index']):
 if c['scene_token']=='de943e246dad4ad686de98008a634ecf':continue
 front=[c['groups'][1][i*3] for i in range(8)]
 for instance in sorted(c['eligible_instances']):
  if not all(instance in r['instance_tokens'] for r in c['groups'][1]):continue
  metrics=[];reason='pass'
  for frame,r in enumerate(front):
   j=list(r['instance_tokens']).index(instance)
   label=r['ori_labels3d'][j];label='.'.join(label) if isinstance(label,(list,tuple)) else str(label)
   if label!='vehicle.bus.rigid':reason='not_rigid_bus';break
   corners=boxes3d_utils.boxes3d_to_corners3d(np.asarray(r['boxes3d'][j:j+1]),rot_axis=1)
   if float(corners[...,2].min())<=.5:reason='near_plane';break
   pts=boxes3d_utils.corners3d_to_corners2d(corners,cam_intrinsic=r['calib']['cam_intrinsic'])[0]
   hull=cv2.convexHull(pts.astype(np.float32));area=cv2.contourArea(hull);inside,_=cv2.intersectConvexConvex(hull,visible_image)
   metric={'frame':frame,'box_index':j,'frame_idx':r['frame_idx'],'area_fraction':float(inside/(1600*900)),
    'visible_fraction':float(inside/area) if area else 0,'box3d':np.asarray(r['boxes3d'][j]).tolist(),'projected_corners':pts.tolist()}
   metrics.append(metric)
   if metric['visible_fraction']<.95:reason='partially_outside';break
   if metric['area_fraction']<.025:reason='small_projection';break
  ledger.append({'candidate_index':c['candidate_index'],'instance_token':instance,'status':reason,'metrics':metrics})
  if reason=='pass':
   selected=copy.deepcopy(c);selected['instance_token']=instance;selected['native_visibility']=metrics;admitted.append(selected);break

save(OUT/'selection_ledger.json',{'visited_instance_windows':len(ledger),'windows':ledger,'selected_candidate':selected['candidate_index'] if selected else None})

(OUT/'admitted.pkl').write_bytes(pickle.dumps(admitted,protocol=4))
print('VISIBLE_CANDIDATES',[(c['candidate_index'],c['scene_token'],c['instance_token']) for c in admitted],flush=True)
