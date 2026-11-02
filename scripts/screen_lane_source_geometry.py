"""Bounded CPU admission of shortlisted windows; no generator/detector scores."""
import copy,hashlib,json,pickle,time
from pathlib import Path
import numpy as np
from shapely.geometry import LineString,Point,Polygon,MultiPoint
from shapely.ops import unary_union
from nuscenes.map_expansion.map_api import NuScenesMap
from dreamer_datasets import boxes3d_utils
from driveloop.ground_alignment import align_to_local_vehicle_bottom
from driveloop.ego_injection import ego_entry_to_cam_box9
from driveloop.plan_geometry import check_plan_geometry
R=Path.cwd();O=R/'outputs/revision_20261006_source_screen';pool=pickle.loads((O/'shortlist.pkl').read_bytes());scenes={r['token']:r for r in json.loads(Path('/data/nuscenes/v1.0-trainval/scene.json').read_text())};logs={r['token']:r for r in json.loads(Path('/data/nuscenes/v1.0-trainval/log.json').read_text())};maps={}
# Freeze selection before map/geometry outcomes are examined.
protocol={'window_order':[c['candidate_index'] for c in pool],'first_passing_window':True,'candidate_order':'headway 16,20,24,28,32 m then speed 4,6,8,10,12 m/s','max_candidates_per_window':25,'minimum_clearance_m':.5,'minimum_endpoint_lane_fraction':.95,'minimum_road_coverage':.99,'wall_budget_seconds_per_invocation':45,'no_video_score_used':True,'six_view_obstacles_all_classes':True,'checkpoint_resume':True}
(O/'geometry_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
checkpoint=O/'geometry_screen.json';ledger=json.loads(checkpoint.read_text())['windows'] if checkpoint.exists() else [];done={c['candidate_index'] for c in ledger};start=time.monotonic()
def poly(box,calib):
 cs=boxes3d_utils.boxes3d_to_corners3d(np.asarray(box,dtype=np.float32)[None],rot_axis=1)[0];T=np.asarray(calib['ego2global'])@calib['cam2ego'];xy=(np.c_[cs,np.ones(8)]@T.T)[:,:2];return MultiPoint(xy).convex_hull

def inspect(c):
 groups=c['groups'];front=[groups[1][i*3] for i in range(8)];location=logs[scenes[c['scene_token']]['log_token']]['location']
 if location not in maps:maps[location]=NuScenesMap(dataroot='/data/nuscenes',map_name=location)
 m=maps[location];egos=[np.array(r['calib']['ego2global'])[:2,3] for r in front];lanes=[m.record_on_point(*p,'lane') for p in egos]
 result={'candidate_index':c['candidate_index'],'scene_token':c['scene_token'],'description':c['scene_description'],'location':location,'ego_lane_tokens':lanes,'trials':[]}
 if '' in lanes or len(set(lanes))!=1:return {**result,'status':'ego_lane_not_stable_or_connector'}
 right=lanes[0]
 if right not in m.arcline_path_3:return {**result,'status':'target_centerline_missing'}
 rightline=LineString(np.asarray(m.discretize_lanes([right],.1)[right])[:,:2]);sbase=rightline.project(Point(egos[0]));anchor=rightline.interpolate(min(sbase+20,rightline.length-1));s=rightline.project(anchor);tangent=np.array(rightline.interpolate(s+.1).coords[0])-np.array(rightline.interpolate(s-.1).coords[0]);tangent/=np.linalg.norm(tangent);normal=np.array([-tangent[1],tangent[0]])
 lefts=[]
 for offset in (3.,4.,5.):
  p=np.array(anchor.coords[0])+offset*normal;token=m.record_on_point(*p,'lane')
  if token and token!=right and token not in lefts:lefts.append(token)
 if len(lefts)!=1:return {**result,'status':'left_lane_missing_or_ambiguous','left_lane_candidates':lefts}
 left=lefts[0]
 if left not in m.arcline_path_3:return {**result,'status':'source_centerline_missing'}
 leftline=LineString(np.asarray(m.discretize_lanes([left],.1)[left])[:,:2]);z=leftline.project(anchor);lt=np.array(leftline.interpolate(z+.1).coords[0])-np.array(leftline.interpolate(z-.1).coords[0]);lt/=np.linalg.norm(lt)
 if float(lt@tangent)<.8:return {**result,'status':'left_lane_opposite_direction'}
 lp=m.extract_polygon(m.get('lane',left)['polygon_token']);rp=m.extract_polygon(m.get('lane',right)['polygon_token']);road=unary_union([lp,rp]);ts=np.array([r['cam_timestamp'] for r in front],dtype=np.int64);ts=(ts-ts[0])/1e6;duration=float(ts[-1]);obstacles=[]
 for i in range(8):
  instances={}
  for group in groups:
   r=group[i*3]
   for j,(b,inst) in enumerate(zip(r['boxes3d'],r['instance_tokens'])):
    p=poly(b,r['calib']);instances.setdefault(inst or f"{r['cam_type']}:{j}",[]).append(p)
  obstacles.append({k:unary_union(v).convex_hull for k,v in instances.items()})
 result.update(source_lane=left,target_lane=right,duration_seconds=duration)
 for headway in (16.,20.,24.,28.,32.):
  for speed in (4.,6.,8.,10.,12.):
   trial={'headway_m':headway,'speed_mps':speed};result['trials'].append(trial)
   if sbase+headway+speed*duration>=rightline.length-5:trial['status']='insufficient_lane_length';continue
   def path(t):
    p=np.array(rightline.interpolate(sbase+headway+speed*t).coords[0]);q=np.array(leftline.interpolate(leftline.project(Point(p))).coords[0]);u=t/duration;w=10*u**3-15*u**4+6*u**5;return q*(1-w)+p*w
   def direction(t):
    d=path(min(duration,t+.002))-path(max(0.,t-.002));return d/np.linalg.norm(d)
   def footprint(t):
    f=direction(t);l=np.array([-f[1],f[0]]);p=path(t);return Polygon([p+x*5*f+y*1.3*l for x,y in ((-1,-1),(1,-1),(1,1),(-1,1))])
   footprints=[footprint(t) for t in ts]
   if footprints[0].intersection(lp).area/footprints[0].area<.95 or footprints[-1].intersection(rp).area/footprints[-1].area<.95:trial['status']='endpoint_lane_coverage';continue
   if min(p.distance(o) for p,frame in zip(footprints,obstacles) for o in frame.values())<.5:trial['status']='sampled_obstacle_clearance';continue
   sweeps=[];road_ok=True
   for i in range(7):
    tt=np.linspace(ts[i],ts[i+1],max(2,int(np.ceil((ts[i+1]-ts[i])/.02))+1));ff=[footprint(t) for t in tt]
    if any(p.intersection(road).area/p.area<.99 for p in ff):road_ok=False;break
    sweep=unary_union(ff).convex_hull;obs=[unary_union([d[k] for d in (obstacles[i],obstacles[i+1]) if k in d]).convex_hull for k in set(obstacles[i])|set(obstacles[i+1])];sweeps.append((sweep,obs))
   if not road_ok:trial['status']='outside_lane_union';continue
   valid=check_plan_geometry(footprints,lp,rp,[list(d.values()) for d in obstacles],swept_pairs=sweeps)
   if not valid['geometry_preflight_passed']:trial['status']='interval_clearance';trial['validation']=valid;continue
   entries=[]
   for i,t in enumerate(ts):
    ref=np.array(front[i]['calib']['ego2global']);g=np.r_[path(t),0.,1.];local=np.linalg.inv(ref)@g;f=direction(t);d=ref[:3,:3].T@np.r_[f,0.];identities=[{k:group[i*3][k] for k in ('cam_type','frame_idx','sample_token','scene_token')} for group in groups]
    entries.append({'relative_frame_idx':i,'frame_idx':int(front[i]['frame_idx']),'actor_id':'actor_00','synthetic_track_id':'actor_00_map_lane_diagnostic','category':'bus','ego':{'center_ego':local[:3].tolist(),'dims':[2.6,3.2,10.],'heading_ego':float(np.arctan2(d[1],d[0]))},'ref_ego2global':ref.tolist(),'sample_identities':identities,'source_record_indices':{group[0]['cam_type']:j*24+i*3 for j,group in enumerate(groups)},'source':'map_lane_source_screen','maneuver':'cut_in','heading':{'mode':'map_quintic_tangent_global'}})
   try:entries,ground=align_to_local_vehicle_bottom(entries,[{'relative_step':i,'record':r} for i,r in enumerate(front)])
   except ValueError as exc:trial['status']='ground_unverified';trial['reason']=str(exc);continue
   actual=[]
   for i,e in enumerate(entries):
    r=front[i];box=ego_entry_to_cam_box9(e['ego'],e['ref_ego2global'],r['calib']['cam2ego'],r['calib']['ego2global']);actual.append(poly(box,r['calib']))
   actual_valid=check_plan_geometry(actual,lp,rp,[list(d.values()) for d in obstacles],swept_pairs=sweeps)
   if not actual_valid['geometry_preflight_passed']:trial['status']='runtime_projection_geometry_failed';trial['validation']=actual_valid;continue
   fine=np.linspace(0,duration,181);xy=np.array([path(t) for t in fine]);vel=np.gradient(xy,fine,axis=0);acc=np.gradient(vel,fine,axis=0)
   trial.update(status='geometry_admitted',validation=actual_valid);result.update(status='geometry_admitted',selected_trial=trial,ground_alignment=ground,approx_peak_acceleration_mps2=float(np.linalg.norm(acc,axis=1).max()),vehicle_dynamics_validated=False)
   (O/'selected_entries.json').write_text(json.dumps(entries,indent=2)+'\n');(O/'selected_window.pkl').write_bytes(pickle.dumps(c,protocol=4));(O/'selected_window.json').write_text(json.dumps({k:v for k,v in c.items() if k!='groups'},indent=2)+'\n');return result
 return {**result,'status':'no_admissible_tested_plan'}

for c in pool:
 if c['candidate_index'] in done:continue
 if time.monotonic()-start>45:break
 r=inspect(c);ledger.append(r);print(c['candidate_index'],r['status'],flush=True)
 checkpoint.write_text(json.dumps({'windows':ledger,'checked':len(ledger),'shortlisted':len(pool),'admitted':any(w['status']=='geometry_admitted' for w in ledger),'seconds_this_invocation':time.monotonic()-start},indent=2)+'\n')
 if r['status']=='geometry_admitted':break
print('SCREEN_CHECKPOINT',len(ledger),time.monotonic()-start,flush=True)
