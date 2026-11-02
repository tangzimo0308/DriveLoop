"""Freeze a map-guided diagnostic plan before looking at generated-video scores."""
import copy,json,pickle,hashlib
from pathlib import Path
import numpy as np
from shapely.geometry import LineString,Point,Polygon,MultiPoint
from shapely.ops import unary_union
from nuscenes.map_expansion.map_api import NuScenesMap
from dreamer_datasets import boxes3d_utils
from driveloop.ego_injection import ego_entry_to_cam_box9
from driveloop.ground_alignment import align_to_local_vehicle_bottom
R=Path.cwd();OUT=R/'outputs/revision_20261006_lane_plan';OUT.mkdir(exist_ok=True)
run=R/'outputs/bus_image_condition_pair_20261006_123610_72403';data=json.loads((run/'summary.json').read_text());meta=next(c for c in data['cases'] if c['actor_injection'])['generation']['metadata'];old=meta['dd2_override_json'];entries=old['boxes3d']['per_frame_append_ego'];lp=Path(meta['dd2_source_sample_binding']['labels_path']);records=pickle.loads(lp.read_bytes())
m=NuScenesMap(dataroot='/data/nuscenes',map_name='singapore-onenorth');left='019c6ebf-278f-482e-ad63-f53c3bbfb7f7';right='034011e2-8169-4f13-9c97-dec18df450b8';lines={t:LineString(np.array(m.discretize_lanes([t],.1)[t])[:,:2]) for t in (left,right)};polys={t:m.extract_polygon(m.get('lane',t)['polygon_token']) for t in (left,right)}
ts=np.array([records[e['source_record_indices']['cam_front']]['cam_timestamp'] for e in entries],dtype=np.int64);ts=(ts-ts[0])/1e6;duration=ts[-1]
c0=(np.array(entries[0]['ref_ego2global'])@np.r_[entries[0]['ego']['center_ego'],1])[:2];s0=lines[right].project(Point(c0));v=10.
def path(t):
 s=s0+v*t
 if s<0 or s>lines[right].length:raise ValueError('outside lane centerline')
 p=np.array(lines[right].interpolate(s).coords[0]);q=np.array(lines[left].interpolate(lines[left].project(Point(p))).coords[0]);u=t/duration;w=10*u**3-15*u**4+6*u**5
 return q*(1-w)+p*w

def tangent(t):
 lo=max(0.,t-.002);hi=min(duration,t+.002);d=path(hi)-path(lo);return d/np.linalg.norm(d)
def foot(t):
 p=path(t);f=tangent(t);l=np.array([-f[1],f[0]])
 return Polygon([p+u*5*f+w*1.3*l for u,w in ((-1,-1),(1,-1),(1,1),(-1,1))])
def source_polygons(e):
 obs={}
 for cam,idx in e['source_record_indices'].items():
  r=records[idx];T=np.array(r['calib']['ego2global'])@r['calib']['cam2ego']
  for j,(box,lab,inst) in enumerate(zip(r['boxes3d'],r['ori_labels3d'],r['instance_tokens'])):
   # Include all available boxes, including pedestrians and barriers.
   cs=boxes3d_utils.boxes3d_to_corners3d(box[None],rot_axis=1)[0];g=(np.c_[cs,np.ones(8)]@T.T)[:,:2];p=MultiPoint(g).convex_hull
   if p.area>0:obs.setdefault(inst or f'{cam}:{j}',[]).append(p)
 return {k:unary_union(v).convex_hull for k,v in obs.items()}
obstacles=[source_polygons(e) for e in entries]

base_station=s0;results=[]
for delta in range(-8,17,2):
 for speed in (4.,6.,8.,10.,12.):
  s0=base_station+delta;v=speed
  clearance=min(foot(t).distance(o) for i,t in enumerate(ts) for o in obstacles[i].values())
  endfrac=foot(duration).intersection(polys[right]).area/foot(duration).area
  startfrac=foot(0).intersection(polys[left]).area/foot(0).area
  swept_min=1e9
  if clearance>=.5 and endfrac>=.95 and startfrac>=.95:
   for i in range(len(ts)-1):
    samples=np.linspace(ts[i],ts[i+1],max(2,int(np.ceil((ts[i+1]-ts[i])/.02))+1));sweep=unary_union([foot(t) for t in samples]).convex_hull
    for k in set(obstacles[i])|set(obstacles[i+1]):
     other=unary_union([d[k] for d in (obstacles[i],obstacles[i+1]) if k in d]).convex_hull;swept_min=min(swept_min,sweep.distance(other))
  else:swept_min=None
  results.append({'station_offset_m':delta,'speed_mps':speed,'minimum_discrete_clearance_m':clearance,'start_lane_fraction':startfrac,'end_lane_fraction':endfrac,'minimum_swept_clearance_m':swept_min,'passed':swept_min is not None and swept_min>=.5})
(OUT/'candidate_grid.json').write_text(json.dumps({'candidate_count':len(results),'selection_uses_video_scores':False,'clearance_m':.5,'endpoint_lane_fraction':.95,'candidates':results},indent=2)+'\n')
print('GRID',len(results),'passed',[x for x in results if x['passed']]);print('DISCRETE_PASS',[x for x in results if x['minimum_discrete_clearance_m']>=.5 and x['start_lane_fraction']>=.95 and x['end_lane_fraction']>=.95])
