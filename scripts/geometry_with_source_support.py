"""Source-only admission of a smoothly advanced native bus track. No video generation."""
import copy,json,pickle
from pathlib import Path
import numpy as np,cv2
from shapely.geometry import MultiPoint,Point
from shapely.ops import unary_union
from nuscenes.map_expansion.map_api import NuScenesMap
from dreamer_datasets import boxes3d_utils,load_dataset
from PIL import Image,ImageDraw
from scripts.run_bus_height_pair import sha,save

import lmdb,hashlib
from driveloop.source_sample_binding import build_source_sample_binding

ROOT=Path('/home/lesliton/projects/DriveLoop')

def copy_native(selected,OUT,protocol):
 OUT.mkdir(parents=True,exist_ok=False)
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
  'visibility':c['native_visibility'],'selection_protocol':protocol,'visited_instance_windows':None,'no_generation_yet':True}
 save(OUT/'manifest.json',manifest)

 return manifest

def check_geometry(m,OUT,signed_delay,source_track):
 OUT.mkdir(parents=True,exist_ok=False);records=pickle.loads((Path(m['dataset'])/'labels/data.pkl').read_bytes());token=m['instance_token'];front=records[24:48]
 assert all(token in r['instance_tokens'] for r in source_track)
 assert len(source_track)>=24
 for a,b in zip(front,source_track[:24]):
  assert a['cam_timestamp']==b['cam_timestamp'] and a['scene_token']==b['scene_token'] and np.array_equal(a['boxes3d'],b['boxes3d'])
 assert all(r['scene_token']==m['scene_token'] and r['cam_type']==front[0]['cam_type'] for r in source_track)
 T=lambda r:np.asarray(r['calib']['ego2global'],np.float64)@np.asarray(r['calib']['cam2ego'],np.float64)
 ts=np.array([r['cam_timestamp'] for r in source_track],np.float64)/1e6;ts-=ts[0];duration=float(ts[21])
 if not (1.7<duration<1.8 and np.all(np.diff(ts)>0) and np.all(np.diff(ts)<=.12)):
  save(OUT/'geometry_checks.json',{'passed':False,'dynamics_validated':False,'trials':[{'passed':False,'reasons':['unexpected_source_timestamp_cadence'],'duration':duration,'times':ts.tolist()}]});return None
 for idx,r in enumerate(records):
  if idx%24 not in range(3,22,3) or token not in r['instance_tokens'] or signed_delay==0:continue
  time=(r['cam_timestamp']-front[0]['cam_timestamp'])/1e6;u=np.clip(time/duration,0,1);phase=max(0,time-signed_delay*(10*u**3-15*u**4+6*u**5))
  if phase>ts[-1]+1e-6:
   save(OUT/'geometry_checks.json',{'passed':False,'dynamics_validated':False,'trials':[{'passed':False,'reasons':['insufficient_source_time_support'],'record_index':idx,'phase':phase,'source_end':float(ts[-1])}]});return None
 centers=[];headings=[]
 for r in source_track:
  b=r['boxes3d'][r['instance_tokens'].index(token)];mat=T(r);centers.append((mat@np.r_[b[:3],1])[:3]);headings.append(mat[:3,:3]@np.array([np.sin(b[7]),0,np.cos(b[7])]))
 centers=np.array(centers);headings=np.array(headings)

 def advanced_pose(r,t,delay):
  j=r['instance_tokens'].index(token);original=r['boxes3d'][j];out=original.copy()
  if t<=0:return out
  u=np.clip(t/duration,0,1);progress=10*u**3-15*u**4+6*u**5;phase=max(0,t-delay*progress)
  assert phase<=ts[-1]+1e-6
  xyz=np.array([np.interp(phase,ts,centers[:,a]) for a in range(3)])
  direction=np.array([np.interp(phase,ts,headings[:,a]) for a in range(3)])
  inverse=np.linalg.inv(T(r));local=inverse@np.r_[xyz,1];h=inverse[:3,:3]@direction
  out[:3]=local[:3];out[7]=np.arctan2(h[0],h[2]);return out

 def footprint(b,r):
  corners=boxes3d_utils.boxes3d_to_corners3d(np.asarray(b)[None],rot_axis=1)[0]
  return MultiPoint((np.c_[corners,np.ones(8)]@T(r).T)[:,:2]).convex_hull

 def project(b,r):
  cs=boxes3d_utils.boxes3d_to_corners3d(np.asarray(b)[None],rot_axis=1)
  if cs[...,2].min()<=.5:return None
  return boxes3d_utils.corners3d_to_corners2d(cs,cam_intrinsic=r['calib']['cam_intrinsic'])[0]
 scenes={x['token']:x for x in json.loads(Path('/data/nuscenes/v1.0-trainval/scene.json').read_text())};logs={x['token']:x for x in json.loads(Path('/data/nuscenes/v1.0-trainval/log.json').read_text())}
 location=logs[scenes[m['scene_token']]['log_token']]['location'];nmap=NuScenesMap(dataroot='/data/nuscenes',map_name=location)
 road=unary_union([nmap.extract_polygon(p) for x in nmap.drivable_area for p in x['polygon_tokens']])
 obstacles=[]
 for i in range(8):
  grouped={}
  for cam in range(6):
   r=records[cam*24+i*3]
   for j,(b,inst) in enumerate(zip(r['boxes3d'],r['instance_tokens'])):
    if inst==token:continue
    grouped.setdefault(inst or f'{cam}:{j}',[]).append(footprint(b,r))
  obstacles.append({k:unary_union(v).convex_hull for k,v in grouped.items()})
 protocol={'candidate_index':m['candidate_index'],'instance_token':token,'trial_delays_seconds':[signed_delay],
  'selection':'source-only fixed amplitude; same original sampled records; extended observed trajectory for interpolation; no generated scores','first_frame_unchanged':True,'only_sampled_records_modified':True,'phase_sign':'phase = t - signed_delay * quintic; negative delay advances',
  'available_future_seconds':float(ts[-1]-duration),'source_track_records':len(source_track),'reference_delay_seconds':None,'equal_magnitude_to_prior_delay':False,
  'interpolation':'linear source global position/direction interpolation; smooth quintic progression of temporal delay',
  'checks':{'minimum_known_obstacle_clearance_m':.5,'minimum_road_fraction':.99,'minimum_visible_projection_fraction':.95,
  'maximum_bottom_change_m':.15,'interval_max_step_s':.02},'claim':'Pose-conditioning diagnostic, not a physically validated driving controller.'}
 save(OUT/'selection_protocol.json',protocol)
 trials=[];chosen=None;viewport=np.array([[0,0],[1600,0],[1600,900],[0,900]],np.float32)
 for delay in protocol['trial_delays_seconds']:
  rr=copy.deepcopy(records)
  for i,r in enumerate(rr):
   if token in r['instance_tokens'] and i%24 in range(3,22,3) and delay!=0:
    t=(r['cam_timestamp']-front[0]['cam_timestamp'])/1e6;r['boxes3d'][r['instance_tokens'].index(token)]=advanced_pose(r,t,delay)
  metrics=[];reason=[];sweeps=[]
  for i in range(8):
   raw=records[24+i*3];r=rr[24+i*3];j=r['instance_tokens'].index(token);b=r['boxes3d'][j];p=footprint(b,r);pts=project(b,r)
   if pts is None:reason.append('near_plane');break
   hull=cv2.convexHull(pts.astype(np.float32));area=cv2.contourArea(hull);inside,_=cv2.intersectConvexConvex(hull,viewport)
   clearance=min((p.distance(o) for o in obstacles[i].values()),default=-1);coverage=p.intersection(road).area/p.area
   before=boxes3d_utils.boxes3d_to_corners3d(raw['boxes3d'][j:j+1],rot_axis=1)[0];after=boxes3d_utils.boxes3d_to_corners3d(np.asarray(b)[None],rot_axis=1)[0]
   z0=(np.c_[before,np.ones(8)]@T(raw).T)[:,2].min();z1=(np.c_[after,np.ones(8)]@T(r).T)[:,2].min()
   rawpts=project(raw['boxes3d'][j],raw);metrics.append({'step':i,'delay_seconds':delay*(10*(ts[i*3]/duration)**3-15*(ts[i*3]/duration)**4+6*(ts[i*3]/duration)**5),
     'clearance_m':clearance,'road_fraction':coverage,'visible_fraction':inside/area,'bottom_change_m':float(z1-z0),
     'projected_corner_mean_shift_x_px_at448':float((pts[:,0].mean()-rawpts[:,0].mean())*448/1600),
     'projected_center_shift_x_px_at448':float(((pts[:,0].min()+pts[:,0].max())-(rawpts[:,0].min()+rawpts[:,0].max()))/2*448/1600),
     'original_box':raw['boxes3d'][j].tolist(),'edited_box':b.tolist()})
   if clearance<.5:reason.append('sampled_clearance')
   if coverage<.99:reason.append('outside_drivable_area')
   if inside/area<.95:reason.append('projection_clipping')
   if abs(z1-z0)>.15:reason.append('bottom_height_change')
  for i in range(7):
   raw=records[24+i*3];times=np.linspace(ts[i*3],ts[(i+1)*3],max(2,int(np.ceil((ts[(i+1)*3]-ts[i*3])/.02))+1))
   polygons=[]
   for time in times:
    b=advanced_pose(raw,time,delay);p=footprint(b,raw);polygons.append(p)
    if p.intersection(road).area/p.area<.99:reason.append('interval_outside_drivable_area')
   sweep=unary_union(polygons).convex_hull
   obs=[unary_union([frame[k] for frame in (obstacles[i],obstacles[i+1]) if k in frame]).convex_hull for k in set(obstacles[i])|set(obstacles[i+1])]
   gap=min((sweep.distance(o) for o in obs),default=-1);sweeps.append({'interval':i,'clearance_m':gap})
   if gap<.5:reason.append('interval_clearance')
  trial={'delay_seconds':delay,'passed':not reason,'reasons':sorted(set(reason)),'frames':metrics,'intervals':sweeps};trials.append(trial)
  if not reason:chosen=rr;break
 save(OUT/'geometry_checks.json',{'trials':trials,'passed':chosen is not None,'dynamics_validated':False})
 if chosen is None:return None
 dataset=OUT/'edited_dataset';dataset.mkdir();(dataset/'labels').mkdir()
 (dataset/'labels/data.pkl').write_bytes(pickle.dumps(chosen,protocol=4))
 for filename in ('config.json','labels/config.json'):(dataset/filename).write_bytes((Path(m['dataset'])/filename).read_bytes())
 for part in ('images','hdmaps'):(dataset/part).symlink_to(Path(m['dataset'])/part,target_is_directory=True)
 changed=[]
 for i,(a,b) in enumerate(zip(records,chosen)):
  other=copy.deepcopy(b);other['boxes3d']=a['boxes3d'];assert pickle.dumps(other,protocol=4)==pickle.dumps(a,protocol=4)
  diff=np.any(a['boxes3d']!=b['boxes3d'],axis=1)
  for j in np.flatnonzero(diff):
   assert a['instance_tokens'][j]==token;assert np.array_equal(a['boxes3d'][j,3:7],b['boxes3d'][j,3:7]);assert a['boxes3d'][j,8]==b['boxes3d'][j,8]
   changed.append({'record_index':i,'camera':a['cam_type'],'frame_idx':a['frame_idx'],'instance_token':token})
 assert np.array_equal(records[24]['boxes3d'],chosen[24]['boxes3d'])
 assert all(x['record_index']%24 in range(3,22,3) for x in changed)
 for cam in range(6):assert np.array_equal(records[cam*24]['boxes3d'],chosen[cam*24]['boxes3d'])
 manifest={'original_manifest':m,'native_dataset':m['dataset'],'edited_dataset':str(dataset),'instance_token':token,
  'selected_delay_seconds':trials[-1]['delay_seconds'],'geometry':trials[-1],'changed_records':changed,
  'native_labels_sha256':sha(Path(m['dataset'])/'labels/data.pkl'),'edited_labels_sha256':sha(dataset/'labels/data.pkl'),
  'metadata_scope':'Only target boxes3d center/yaw changed. Original source metadata such as velocities/corners remains source provenance and is not edited-video ground truth; DD2 consumes boxes3d for rasterization.'}
 save(OUT/'manifest.json',manifest)

 return manifest
