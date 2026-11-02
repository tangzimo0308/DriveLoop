"""Expanded source-only inventory, fixed admission rules, no generation or detector."""
import json,pickle,copy,time,hashlib
from pathlib import Path
from collections import Counter
import numpy as np,cv2
from dreamer_datasets import boxes3d_utils
from scripts.run_dd2_batch_sampler_audit import candidate_camera_starts
ROOT=Path('/home/lesliton/projects/DriveLoop');OUT=ROOT/'outputs/revision_20261019_val_source_projection';SOURCE=Path('/data/projects/DriveLoop/data/processed/nuscenes/cam_all_val/v0.0.2/labels/data.pkl');HIST=ROOT/'outputs/revision_20261019_history_inventory/inventory.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
OUT.mkdir(exist_ok=False);start=time.monotonic();history=json.loads(HIST.read_text());assert not history['parse_errors'] and not history['unresolved_generation_records']
oldindices=set()
plan_path=ROOT/'outputs/revision_20261019_measurable_source_plan/protocol.json'
plan=json.loads(plan_path.read_text());assert plan['source_dataset']==str(SOURCE.parent.parent) and plan['history_sha256']==sha(HIST)
protocol={'amended_plan_sha256':sha(plan_path),'source_split':'cam_all_val_v0.0.2','candidate_identity':'dataset path + index + scene + instance; never join train and val by bare candidate index','stage':'source projection and temporal support inventory; geometry and DD2 admission NOT performed','source':str(SOURCE),'source_sha256':sha(SOURCE),'source_size':SOURCE.stat().st_size,'source_mtime_ns':SOURCE.stat().st_mtime_ns,'history_sha256':sha(HIST),'script_sha256':sha(Path(__file__)),'excluded_scenes':sorted(history['excluded_scenes']),'candidate_order':'all source candidate indices ascending; all rigid bus instances lexicographic; no two-window cap','frame_num':8,'stride':3,'track_cap':40,'max_gap_s':.12,'visible_fraction_min':.95,'area_fraction_min':.025,'depth_min_m':.5,'initial_phase_s':[-.25,.25],'initial_absolute_late_signal_min_px':8,'late_sign_consistency_required':True,'source_curve_grid_s':[-.5,-.375,-.25,-.125,0,.125,.25,.375,.5],'range_requirement':'all grid projections visible and strictly monotone mean response; no extrapolation','pending_geometry_selection':'First range-qualified window/instance per scene; first six scenes by candidate index; all failures retained. Not yet generation approval.','limits':['Development-unseen only against recorded history, not certified unexposed to generator training.','Visibility and signal selected pool is not an unbiased dataset sample.','Projection tests do not check occlusion, road geometry, obstacles, dynamics or generated appearance.'],'generation_run':False}
save(OUT/'protocol.json',protocol);print('LOADING_LABELS',flush=True)
records=pickle.loads(SOURCE.read_bytes());starts=candidate_camera_starts(records,frame_num=8,hz_factor=3,video_split_rate=1,multiview=True)
print('LOADED',len(records),len(starts),round(time.monotonic()-start,2),flush=True)
viewport=np.array([[0,0],[1600,0],[1600,900],[0,900]],np.float32)
def project(b,r):
 cs=boxes3d_utils.boxes3d_to_corners3d(np.asarray(b)[None],rot_axis=1)
 if cs[...,2].min()<=.5:return None
 pts=boxes3d_utils.corners3d_to_corners2d(cs,cam_intrinsic=r['calib']['cam_intrinsic'])[0];h=cv2.convexHull(pts.astype(np.float32));area=cv2.contourArea(h);inside,_=cv2.intersectConvexConvex(h,viewport)
 return {'cx':float((pts[:,0].min()+pts[:,0].max())/2*448/1600),'visible_fraction':float(inside/area) if area else 0,'area_fraction':float(inside/(1600*900))}
def visible(p):return p is not None and p['visible_fraction']>=.95 and p['area_fraction']>=.025
T=lambda r:np.asarray(r['calib']['ego2global'],float)@np.asarray(r['calib']['cam2ego'],float)
def label(v):return '.'.join(v) if isinstance(v,(list,tuple)) else str(v)
def inspect(front,token,idx):
 ts0=front[0]['cam_timestamp'];duration=(front[21]['cam_timestamp']-ts0)/1e6
 row={'instance_token':token,'duration_s':duration,'initial_signal_pass':False,'range_projection_pass':False}
 if not 1.7<duration<1.8:return {**row,'status':'unexpected_cadence'},None
 track=[]
 for j in range(40):
  if idx+j>=len(records):break
  r=records[idx+j]
  if r['scene_token']!=front[0]['scene_token'] or r['cam_type']!=front[0]['cam_type'] or token not in r['instance_tokens']:break
  if track and not 0<(r['cam_timestamp']-track[-1]['cam_timestamp'])/1e6<=.12:break
  track.append(r)
 row['track_records']=len(track)
 if len(track)<24:return {**row,'status':'insufficient_contiguous_track'},None
 ts=np.array([(r['cam_timestamp']-ts0)/1e6 for r in track]);row['future_support_s']=float(ts[-1]-duration)
 if ts[-1]+1e-6<duration+.25:return {**row,'status':'insufficient_initial_future_support'},None
 src=[project(r['boxes3d'][list(r['instance_tokens']).index(token)],r) for r in front[::3][:8]]
 if not all(visible(p) for p in src):return {**row,'status':'original_projection_visibility'},None
 centers=[];headings=[]
 for r in track:
  b=r['boxes3d'][list(r['instance_tokens']).index(token)];mat=T(r);centers.append((mat@np.r_[b[:3],1])[:3]);headings.append(mat[:3,:3]@np.array([np.sin(b[7]),0,np.cos(b[7])]))
 centers=np.array(centers);headings=np.array(headings)
 def phase_projection(delay):
  fs=[]
  for i in range(8):
   r=front[i*3];t=(r['cam_timestamp']-ts0)/1e6;u=t/duration;phase=t-delay*(10*u**3-15*u**4+6*u**5)
   if phase<ts[0]-1e-6 or phase>ts[-1]+1e-6:return {'phase_s':delay,'passed':False,'reason':'future_support','frames':fs}
   b=r['boxes3d'][list(r['instance_tokens']).index(token)].copy()
   if i and delay!=0:
    xyz=np.array([np.interp(phase,ts,centers[:,a]) for a in range(3)]);head=np.array([np.interp(phase,ts,headings[:,a]) for a in range(3)]);inv=np.linalg.inv(T(r));b[:3]=(inv@np.r_[xyz,1])[:3];d=inv[:3,:3]@head;b[7]=np.arctan2(d[0],d[2])
   p=project(b,r)
   if not visible(p):return {'phase_s':delay,'passed':False,'reason':'projection_visibility','failed_frame':i,'frames':fs}
   fs.append({'frame':i,'dx_px':p['cx']-src[i]['cx'],'visible_fraction':p['visible_fraction'],'area_fraction':p['area_fraction']})
  return {'phase_s':delay,'passed':True,'frames':fs,'source_mean_dx_px':float(np.mean([f['dx_px'] for f in fs[4:]]))}
 initial=[phase_projection(d) for d in [-.25,.25]];row['initial']=initial
 if not all(x['passed'] for x in initial):return {**row,'status':'initial_projection_visibility'},None
 means=[x['source_mean_dx_px'] for x in initial]
 if min(map(abs,means))<8 or means[0]*means[1]>=0:return {**row,'status':'insufficient_opposite_signal'},None
 if any(f['dx_px']*x['source_mean_dx_px']<=0 for x in initial for f in x['frames'][4:]):return {**row,'status':'late_sign_inconsistent'},None
 row['initial_signal_pass']=True
 curve=[phase_projection(d) for d in protocol['source_curve_grid_s']];row['curve']=curve
 if not all(x['passed'] for x in curve):return {**row,'status':'range_support_or_visibility'},None
 diff=np.diff([x['source_mean_dx_px'] for x in curve])
 if not (np.all(diff>0) or np.all(diff<0)):return {**row,'status':'nonmonotone_source_curve'},None
 return {**row,'status':'range_projection_qualified','range_projection_pass':True,'source_visibility':src},track
ledger=[];counts=Counter();instance_counts=Counter();selected=[];selected_scenes=set()
for n,group in enumerate(starts):
 front=records[group[1]:group[1]+24];scene=front[0]['scene_token']
 row={'candidate_index':n,'scene_token':scene,'source_split':'cam_all_val_v0.0.2','instances':[]}
 if scene in history['excluded_scenes']:row['status']='excluded_recorded_history'
 elif len(front)!=24 or any(r['scene_token']!=scene or r['cam_type']!=front[0]['cam_type'] for r in front):row['status']='source_window_boundary'
 else:
  eligible=None
  for r in front:
   inst={t for t,l in zip(r['instance_tokens'],r['ori_labels3d']) if t and label(l)=='vehicle.bus.rigid'};eligible=inst if eligible is None else eligible&inst
  if not eligible:row['status']='no_rigid_bus_continuous_24_records'
  else:
   for token in sorted(eligible):
    check,track=inspect(front,token,group[1]);row['instances'].append(check);instance_counts[check['status']]+=1
    if check['range_projection_pass'] and scene not in selected_scenes and len(selected)<6:
     groups=[records[g:g+24] for g in group]
     if not all(len(v)==24 and all(r['scene_token']==scene and r['cam_type']==v[0]['cam_type'] for r in v) for v in groups):
      check['range_projection_pass']=False;check['status']='multiview_window_boundary';continue
     selected_scenes.add(scene);selected.append(copy.deepcopy({'candidate_index':n,'source_dataset':str(SOURCE.parent.parent),'source_split':'cam_all_val_v0.0.2','source_camera_starts':group,'scene_token':scene,'scene_description':front[0]['scene_description'],'instance_token':token,'eligible_instances':[token],'groups':groups,'extended_track':track,'native_visibility':[{'frame':i,**p} for i,p in enumerate(check['source_visibility'])],'source_screen':check}))
   row['status']='range_projection_qualified' if any(x['range_projection_pass'] for x in row['instances']) else 'instance_checks_failed'
 counts[row['status']]+=1;ledger.append(row)
 if (n+1)%1000==0:print('SCREENED',n+1,'selected_scenes',len(selected),'seconds',round(time.monotonic()-start,2),flush=True)
assert SOURCE.stat().st_size==protocol['source_size'] and SOURCE.stat().st_mtime_ns==protocol['source_mtime_ns'];assert sha(HIST)==protocol['history_sha256']
(OUT/'selected_candidates.pkl').write_bytes(pickle.dumps(selected,protocol=4))
summary={'complete_source_inventory':True,'source_records':len(records),'source_windows':len(starts),'counts':dict(counts),'instance_counts':dict(instance_counts),'initial_signal_pass_windows':sum(any(x['initial_signal_pass'] for x in w['instances']) for w in ledger),'range_projection_pass_windows':sum(w['status']=='range_projection_qualified' for w in ledger),'range_projection_pass_scenes':len({w['scene_token'] for w in ledger if w['status']=='range_projection_qualified'}),'selected_for_geometry':[{'candidate_index':c['candidate_index'],'scene_token':c['scene_token'],'instance_token':c['instance_token'],'source_camera_starts':c['source_camera_starts'],'source_split':'cam_all_val_v0.0.2'} for c in selected],'geometry_verified':False,'runtime_input_verified':False,'generation_run':False,'elapsed_seconds':time.monotonic()-start,'selected_pickle_sha256':sha(OUT/'selected_candidates.pkl'),'protocol_sha256':sha(OUT/'protocol.json')}
save(OUT/'window_ledger.json',ledger);save(OUT/'summary.json',summary);print(json.dumps(summary),flush=True)
