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
ROOT=Path('/home/lesliton/projects/DriveLoop');SOURCE=ROOT/'outputs/revision_20261006_native_bus_source';OUT=ROOT/'outputs/revision_20261006_native_advance_source';OUT.mkdir(exist_ok=False)
m=json.loads((SOURCE/'manifest.json').read_text());records=pickle.loads((Path(m['dataset'])/'labels/data.pkl').read_bytes());token=m['instance_token'];front=records[24:48]
assert all(token in r['instance_tokens'] for r in front)
T=lambda r:np.asarray(r['calib']['ego2global'],np.float64)@np.asarray(r['calib']['cam2ego'],np.float64)
ts=np.array([r['cam_timestamp'] for r in front],np.float64)/1e6;ts-=ts[0];duration=float(ts[21]);assert 1.7<duration<1.8
centers=[];headings=[]
for r in front:
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
protocol={'candidate_index':m['candidate_index'],'instance_token':token,'trial_delays_seconds':[-.125,-.0625],
 'selection':'first geometrically admitted advance in fixed order; negative signed delay means advance; no generated scores','first_frame_unchanged':True,'only_sampled_records_modified':True,'phase_sign':'phase = t - signed_delay * quintic; negative delay advances',
 'available_future_seconds':float(ts[-1]-duration),'reference_delay_seconds':.25,'equal_magnitude_to_prior_delay':False,
 'interpolation':'linear source global position/direction interpolation; smooth quintic progression of temporal delay',
 'checks':{'minimum_known_obstacle_clearance_m':.5,'minimum_road_fraction':.99,'minimum_visible_projection_fraction':.95,
 'maximum_bottom_change_m':.15,'interval_max_step_s':.02},'claim':'Pose-conditioning diagnostic, not a physically validated driving controller.'}
save(OUT/'selection_protocol.json',protocol)
trials=[];chosen=None;viewport=np.array([[0,0],[1600,0],[1600,900],[0,900]],np.float32)
for delay in protocol['trial_delays_seconds']:
 rr=copy.deepcopy(records)
 for i,r in enumerate(rr):
  if token in r['instance_tokens'] and i%24 in range(0,22,3):
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
assert chosen is not None,'No advanced native trajectory passed geometry checks; do not generate.'
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
assert len(changed)==7 and all(x['record_index'] in [24+i*3 for i in range(1,8)] for x in changed)
manifest={'original_manifest':m,'native_dataset':m['dataset'],'edited_dataset':str(dataset),'instance_token':token,
 'selected_delay_seconds':trials[-1]['delay_seconds'],'geometry':trials[-1],'changed_records':changed,
 'native_labels_sha256':sha(Path(m['dataset'])/'labels/data.pkl'),'edited_labels_sha256':sha(dataset/'labels/data.pkl'),
 'metadata_scope':'Only target boxes3d center/yaw changed. Original source metadata such as velocities/corners remains source provenance and is not edited-video ground truth; DD2 consumes boxes3d for rasterization.'}
save(OUT/'manifest.json',manifest)
ds=load_dataset(m['dataset']);sheet=Image.new('RGB',(448*4,292*2),'white');draw=ImageDraw.Draw(sheet)
for i in range(8):
 raw=ds[24+i*3];im=raw['image'].convert('RGB').resize((448,252));d=ImageDraw.Draw(im)
 for box,color in [(trials[-1]['frames'][i]['original_box'],'blue'),(trials[-1]['frames'][i]['edited_box'],'red')]:
  pts=project(np.array(box),raw)*[448/1600,252/900]
  for a,b in ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)):d.line([tuple(pts[a]),tuple(pts[b])],fill=color,width=2)
 x=(i%4)*448;y=(i//4)*292;sheet.paste(im,(x,y+36));draw.text((x+5,y+8),f'frame{i} | blue original | red advanced',fill='black')
sheet.save(OUT/'native_advance_plan.jpg',quality=95)
print('NATIVE_ADVANCE_ADMITTED signed_delay',trials[-1]['delay_seconds'],'last-frame shift px',trials[-1]['frames'][-1]['projected_center_shift_x_px_at448'],'changed records',len(changed),flush=True)
