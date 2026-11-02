"""Select a native bus using source labels only, then copy its unchanged window."""
import copy,json,pickle,hashlib
from pathlib import Path
import cv2,numpy as np,lmdb
from dreamer_datasets import boxes3d_utils,load_dataset
from driveloop.source_sample_binding import build_source_sample_binding
from scripts.run_bus_height_pair import sha,save
from PIL import Image,ImageDraw
ROOT=Path('/home/lesliton/projects/DriveLoop');OUT=ROOT/'outputs/revision_20261006_native_bus_source'
OUT.mkdir(exist_ok=False)
cache=ROOT/'outputs/revision_20261006_source_screen_expanded/shortlist.pkl'
shortlist=pickle.loads(cache.read_bytes());assert len(shortlist)==337
protocol={'shortlist':str(cache),'shortlist_sha256':sha(cache),'source_pool':337,'selection':'first candidate in ascending index, then lexicographic instance token, satisfying all rules; source labels only; no generated scores',
 'rules':{'category':'vehicle.bus.rigid','all_8_frames':True,'all_corners_depth_min_m':.5,'projected_visible_fraction_min':.95,'projected_convex_area_fraction_min':.025},
 'pool_limit':'Reuses prior 337-window shortlist, which excluded candidate28 scene and capped 2 windows per scene. Not an unbiased dataset sample.',
 'claim':'Source visibility positive-control selection; not method evaluation or new maneuver generation.'}
save(OUT/'selection_protocol.json',protocol)
ledger=[];selected=None;visible_image=np.array([[0,0],[1600,0],[1600,900],[0,900]],np.float32)
for c in sorted(shortlist,key=lambda x:x['candidate_index']):
 front=[c['groups'][1][i*3] for i in range(8)]
 for instance in sorted(c['eligible_instances']):
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
   selected=copy.deepcopy(c);selected['instance_token']=instance;selected['native_visibility']=metrics;break
 if selected:break
save(OUT/'selection_ledger.json',{'visited_instance_windows':len(ledger),'windows':ledger,'selected_candidate':selected['candidate_index'] if selected else None})
assert selected is not None,'No source met preregistered visibility rules; no generation is permitted.'
c=selected;(OUT/'selected_window.pkl').write_bytes(pickle.dumps(c,protocol=4))
source=Path('/data/projects/DriveLoop/data/processed/nuscenes/cam_all_train/v0.0.2');dataset=OUT/'dataset';dataset.mkdir()
old_to_new={old:i*24 for i,old in enumerate(c['source_camera_starts'])};records=[];source_indices=[]
for group in c['groups']:
 for r in group:
  new=copy.deepcopy(r);source_indices.append(int(new['data_index']));new['data_index']=len(records)
  new['multiview_start_idx']={k:old_to_new[v] for k,v in new.get('multiview_start_idx',{}).items() if v in old_to_new};records.append(new)
(dataset/'labels').mkdir();(dataset/'labels/data.pkl').write_bytes(pickle.dumps(records,protocol=4))
conf=json.loads((source/'labels/config.json').read_text());conf['data_size']=len(records);save(dataset/'labels/config.json',conf)
hashes={}
for component in ('images','hdmaps'):
 (dataset/component).mkdir();src=lmdb.open(str(source/component),readonly=True,lock=False,readahead=False);dst=lmdb.open(str(dataset/component),map_size=2**30)
 hashes[component]=[]
 with src.begin() as read,dst.begin(write=True) as write:
  for i,old in enumerate(source_indices):
   value=read.get(str(old).encode());assert value is not None;write.put(str(i).encode(),value);hashes[component].append(hashlib.sha256(value).hexdigest())
 src.close();dst.close();conf=json.loads((source/component/'config.json').read_text());conf['data_size']=len(records);conf.pop('data_path',None);save(dataset/component/'config.json',conf)
save(dataset/'config.json',{'_class_name':'Dataset','config_paths':['labels/config.json','images/config.json','hdmaps/config.json']})
identity={'candidate':f"candidate{c['candidate_index']}",'provenance':'native rigid-bus source visibility control',
 'frame_summaries':[{k:g[i*3][k] for k in ('cam_type','sample_token','scene_token','frame_idx','cam_token','cam_timestamp')} for g in c['groups'] for i in range(8)]}
save(OUT/'identity.json',identity)
binding=build_source_sample_binding(dataset,source_candidate_id=identity['candidate'],instance_token=c['instance_token'],scene_token=c['scene_token'],identity_summary_path=OUT/'identity.json',frame_num=8,hz_factor=3,video_split_rate=1,multiview=True)
assert binding['ready'],binding
manifest={'candidate_index':c['candidate_index'],'source_candidate_id':identity['candidate'],'scene_token':c['scene_token'],'instance_token':c['instance_token'],
 'scene_description':c['scene_description'],'dataset':str(dataset),'identity':str(OUT/'identity.json'),'source_dataset':str(source),
 'source_camera_starts':c['source_camera_starts'],'source_data_indices':source_indices,'image_and_map_hashes':hashes,'binding':binding,
 'visibility':c['native_visibility'],'selection_protocol':protocol,'visited_instance_windows':len(ledger),'no_generation_yet':True}
save(OUT/'manifest.json',manifest)
loaded=load_dataset(str(dataset));sheet=Image.new('RGB',(448*4,296*2),'white');draw=ImageDraw.Draw(sheet)
for i,metric in enumerate(c['native_visibility']):
 raw=loaded[24+i*3];assert raw['scene_token']==c['scene_token'];assert raw['instance_tokens'][metric['box_index']]==c['instance_token']
 im=raw['image'].convert('RGB').resize((448,252));d=ImageDraw.Draw(im);pts=np.array(metric['projected_corners'])*[448/1600,252/900]
 for a,b in ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)):d.line([tuple(pts[a]),tuple(pts[b])],fill='red',width=2)
 x=(i%4)*448;y=(i//4)*296;sheet.paste(im,(x,y+35));draw.text((x+5,y+8),f'Native source candidate {c["candidate_index"]} | frame {i}',fill='black')
sheet.save(OUT/'native_source_bus.jpg',quality=95)
print('NATIVE_SOURCE_READY',c['candidate_index'],c['scene_description'],'visited',len(ledger),'min area',min(x['area_fraction'] for x in c['native_visibility']),flush=True)
