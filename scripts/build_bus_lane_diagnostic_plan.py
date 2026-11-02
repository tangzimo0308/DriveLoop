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
from driveloop.plan_geometry import check_plan_geometry,require_geometry_ready
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
new=copy.deepcopy(entries);front=[]
for i,(e,t) in enumerate(zip(new,ts)):
 T=np.array(e['ref_ego2global']);g=T@np.r_[e['ego']['center_ego'],1];g[:2]=path(t);local=np.linalg.inv(T)@g;e['ego']['center_ego']=local[:3].tolist();f=tangent(t);d=T[:3,:3].T@np.r_[f,0];e['ego']['heading_ego']=float(np.arctan2(d[1],d[0]));e['heading']={'mode':'map_quintic_path_tangent_global'};e['trajectory_repair']={'mode':'map_lane_diagnostic_v1','source_lane':left,'target_lane':right,'t_seconds':float(t)};front.append({'relative_step':i,'record':records[e['source_record_indices']['cam_front']]})
new,ground=align_to_local_vehicle_bottom(new,front)
rows=[]
for i,(e,t) in enumerate(zip(new,ts)):
 r=front[i]['record'];box=ego_entry_to_cam_box9(e['ego'],e['ref_ego2global'],r['calib']['cam2ego'],r['calib']['ego2global']);cs=boxes3d_utils.boxes3d_to_corners3d(np.array([box],dtype=np.float32),rot_axis=1)[0];T=np.array(r['calib']['ego2global'])@r['calib']['cam2ego'];g=(np.c_[cs,np.ones(8)]@T.T)[:,:2];p=MultiPoint(g).convex_hull
 nearest=min(((p.distance(o),k) for k,o in obstacles[i].items()),default=(float('inf'),''));hits=[k for k,o in obstacles[i].items() if p.intersection(o).area>1e-6]
 rows.append({'step':i,'time_seconds':float(t),'center_global_xy':path(t).tolist(),'target_lane_contains_center':polys[right].covers(Point(path(t))),'footprint_in_target_lane_fraction':p.intersection(polys[right]).area/p.area,'footprint_in_source_lane_fraction':p.intersection(polys[left]).area/p.area,'minimum_source_box_clearance_m':nearest[0],'nearest_source_instance':nearest[1],'intersecting_source_instances':hits,'front_cam_box9':box,'planned_footprint_global':list(p.exterior.coords),'source_instance_count':len(obstacles[i])})
intervals=[];swept_pairs=[]
for i in range(len(ts)-1):
 allids=set(obstacles[i])|set(obstacles[i+1]);swept={k:unary_union([table[k] for table in (obstacles[i],obstacles[i+1]) if k in table]).convex_hull for k in allids}
 # Entire planned center sweep is conservatively enclosed, with 20 ms samples.
 samples=np.linspace(ts[i],ts[i+1],max(2,int(np.ceil((ts[i+1]-ts[i])/.02))+1));bus_sweep=unary_union([foot(t) for t in samples]).convex_hull
 swept_pairs.append((bus_sweep,list(swept.values())))
 nearest=min((bus_sweep.distance(p),k) for k,p in swept.items());intervals.append({'interval':i,'minimum_swept_clearance_m':nearest[0],'nearest_source_instance':nearest[1],'source_visibility_changes':sorted(set(obstacles[i])^set(obstacles[i+1]))})
fine=np.linspace(0,duration,181);xy=np.array([path(t) for t in fine]);vel=np.gradient(xy,fine,axis=0);acc=np.gradient(vel,fine,axis=0)
validation=check_plan_geometry([Polygon(r['planned_footprint_global']) for r in rows],polys[left],polys[right],[list(o.values()) for o in obstacles],swept_pairs=swept_pairs)
geom_ok=validation['geometry_preflight_passed']
report={'schema':'map_lane_diagnostic_v1','geometry_validation':validation,'geometry_preflight_passed':geom_ok,'vehicle_dynamics_validated':False,'semantic_video_success':False,'generation_authorized_by_geometry_alone':False,'source_run':str(run),'source_labels_sha256':hashlib.sha256(lp.read_bytes()).hexdigest(),'map_sha256':hashlib.sha256(Path('/data/nuscenes/maps/expansion/singapore-onenorth.json').read_bytes()).hexdigest(),'source_lane':left,'target_lane':right,'speed_along_lane_mps':v,'start_station_on_target_lane_m':s0,'duration_seconds':float(duration),'selection':'CPU-only exploratory grid: station offsets -8..16 m by 2; speeds 4,6,8,10,12 m/s. Choose zero station offset (preserve progress) and lowest passing speed in that offset; no video scores used. Final stricter six-view/swept audit controls readiness.','footprint_clearance_threshold_m':.5,'endpoint_lane_fraction_threshold':.95,'approx_peak_speed_mps':float(np.linalg.norm(vel,axis=1).max()),'approx_peak_acceleration_mps2':float(np.linalg.norm(acc,axis=1).max()),'frames':rows,'intervals':intervals,'ground_alignment':ground,'limitations':['Dynamics not validated; short 1.8 s transition is aggressive. Numerical acceleration is a diagnostic on a discretized map, not a calibrated vehicle model.','Six-view source annotations at sampled camera timestamps, with conservative endpoint swept hulls; missing/unannotated motion remains unknown.','Static map can differ from construction-zone reality.','This changes longitudinal speed and lane path together; not a single-parameter ablation.']}
(OUT/'plan_audit.json').write_text(json.dumps(report,indent=2)+'\n')
if not geom_ok:
 print('PLAN_REJECTED',[(r['step'],r['minimum_source_box_clearance_m']) for r in rows],intervals);raise SystemExit(2)
require_geometry_ready(report)
result=copy.deepcopy(old);result['boxes3d']['per_frame_append_ego']=new;result['audit']['map_lane_diagnostic']= {k:report[k] for k in ('schema','geometry_preflight_passed','vehicle_dynamics_validated','source_lane','target_lane')};result['audit'].pop('image_condition_probe_actor_injection',None)
(OUT/'candidate_override.json').write_text(json.dumps(result,indent=2)+'\n')
print('GEOMETRY_PASS',json.dumps({k:report[k] for k in ('duration_seconds','approx_peak_speed_mps','approx_peak_acceleration_mps2')}));print('min discrete/swept',min(r['minimum_source_box_clearance_m'] for r in rows),min(r['minimum_swept_clearance_m'] for r in intervals));print('end lane fraction',rows[-1]['footprint_in_target_lane_fraction'])
