"""Source-only bounded future-track and projected-signal inventory. No model loading."""
import json,pickle,copy,time,hashlib
from pathlib import Path
import numpy as np,cv2
from dreamer_datasets import boxes3d_utils
ROOT=Path('/home/lesliton/projects/DriveLoop');OUT=ROOT/'outputs/revision_20261008_signal_precheck';OUT.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,o):p.write_text(json.dumps(o,indent=2)+'\n')
start=time.monotonic();base=ROOT/'outputs/revision_20261007_native_multiscene_source';screen=ROOT/'outputs/revision_20261007_multiscene_screen/admitted.pkl';source=Path('/data/projects/DriveLoop/data/processed/nuscenes/cam_all_train/v0.0.2/labels/data.pkl')
ledger=json.loads((base/'detailed_screening_results.json').read_text());candidates=pickle.loads(screen.read_bytes());old={r['candidate_index']:r for r in ledger}
protocol={'stage':'future-support and signal only; later geometry remains mandatory','input_pool':'16 previously visibility-screened windows; only prior original-geometry passes; exclude previously generated candidate5596 scene (candidate91 scene was already excluded)','shortlist_sha256':sha(screen),'prior_geometry_sha256':sha(base/'detailed_screening_results.json'),'source_size':source.stat().st_size,'source_mtime_ns':source.stat().st_mtime_ns,'amplitudes_seconds':[.25,.5],'future_source_record_cap':40,'generated_frame_count':8,'stride':3,'required_time_gap_max_s':.12,'projected_visible_fraction_min':.95,'projected_convex_area_fraction_min':.025,'required_absolute_late_mean_pixels':8,'required_opposite_signs':True,'require_each_late_frame_same_sign_as_its_mean':True,'signal_threshold_scope':'Exploratory design criterion chosen after previous measurement limitations; not calibrated measurement error, a significance cutoff, or a generation success threshold. No new outputs viewed.','claim':'Source-only projection feasibility; excludes occlusion/appearance assessment, dynamics, runtime adapter and generation validation.','script_sha256':sha(__file__)}
save(OUT/'protocol.json',protocol)
selected=[];eligibility=[]
for c in candidates:
 n=c['candidate_index'];reason='candidate'
 if c['scene_token']=='796e7ae080984a0ab2b96ce80f18ca3b':reason='previously_generated_scene'
 elif old[n]['details']['original']!=[[]]:reason='prior_original_geometry_failed'
 eligibility.append({'candidate_index':n,'scene_token':c['scene_token'],'status':reason})
 if reason=='candidate':selected.append(c)
save(OUT/'eligibility.json',eligibility);print('LOADING_SOURCE_LABELS',len(selected),'eligible windows',flush=True)
records=pickle.loads(source.read_bytes());tracks={};support=[]
for c in selected:
 n=c['candidate_index'];front=c['groups'][1];token=c['instance_token'];idx=c['source_camera_starts'][1];track=[];stop=None
 for j in range(40):
  if idx+j>=len(records):stop='end_of_records';break
  r=records[idx+j]
  if r['cam_type']!=front[0]['cam_type'] or r['scene_token']!=c['scene_token']:stop='camera_or_scene_boundary';break
  if token not in r['instance_tokens']:stop='target_absent';break
  if track and not (0<(r['cam_timestamp']-track[-1]['cam_timestamp'])/1e6<=.12):stop='timestamp_gap';break
  if j<24:assert pickle.dumps(r,protocol=4)==pickle.dumps(front[j],protocol=4)
  track.append(copy.deepcopy(r))
 tracks[n]=track
 support.append({'candidate_index':n,'available_records':len(track),'stop_reason':stop or 'record_cap','source_indices':[int(r['data_index']) for r in track]})
del records
(OUT/'extended_tracks.pkl').write_bytes(pickle.dumps(tracks,protocol=4));save(OUT/'track_support.json',support)
T=lambda r:np.asarray(r['calib']['ego2global'],float)@np.asarray(r['calib']['cam2ego'],float)
viewport=np.array([[0,0],[1600,0],[1600,900],[0,900]],np.float32)
def project(box,r):
 cs=boxes3d_utils.boxes3d_to_corners3d(np.asarray(box)[None],rot_axis=1)
 if cs[...,2].min()<=.5:return None
 pts=boxes3d_utils.corners3d_to_corners2d(cs,cam_intrinsic=r['calib']['cam_intrinsic'])[0];h=cv2.convexHull(pts.astype(np.float32));area=cv2.contourArea(h);inside,_=cv2.intersectConvexConvex(h,viewport)
 return {'cx':float((pts[:,0].min()+pts[:,0].max())/2*448/1600),'visible_fraction':float(inside/area),'area_fraction':float(inside/(1600*900))}
rows=[]
for c in selected:
 n=c['candidate_index'];track=tracks[n];token=c['instance_token'];front=c['groups'][1];ts=np.array([r['cam_timestamp'] for r in track],float)/1e6;ts-=front[0]['cam_timestamp']/1e6;duration=(front[21]['cam_timestamp']-front[0]['cam_timestamp'])/1e6
 centers=[];headings=[]
 for r in track:
  b=r['boxes3d'][r['instance_tokens'].index(token)];mat=T(r);centers.append((mat@np.r_[b[:3],1])[:3]);headings.append(mat[:3,:3]@np.array([np.sin(b[7]),0,np.cos(b[7])]))
 centers=np.array(centers);headings=np.array(headings)
 for amp in protocol['amplitudes_seconds']:
  row={'candidate_index':n,'scene_token':c['scene_token'],'instance_token':token,'amplitude_seconds':amp,'duration':duration,'available_future_seconds':float(ts[-1]-duration) if len(ts) else None,'arms':{},'reasons':[]}
  if not (1.7<duration<1.8) or len(track)<24:row['reasons'].append('insufficient_contiguous_original_support')
  elif ts[-1]+1e-6<duration+amp:row['reasons'].append('insufficient_future_support')
  else:
   for arm,delay in [('delayed',amp),('advanced',-amp)]:
    fs=[]
    for i in range(8):
     r=front[i*3];t=(r['cam_timestamp']-front[0]['cam_timestamp'])/1e6;u=t/duration;phase=t-delay*(10*u**3-15*u**4+6*u**5)
     assert ts[0]-1e-6<=phase<=ts[-1]+1e-6
     j=r['instance_tokens'].index(token);original=r['boxes3d'][j];box=original.copy()
     if i:
      xyz=np.array([np.interp(phase,ts,centers[:,a]) for a in range(3)]);direction=np.array([np.interp(phase,ts,headings[:,a]) for a in range(3)]);inv=np.linalg.inv(T(r));box[:3]=(inv@np.r_[xyz,1])[:3];h=inv[:3,:3]@direction;box[7]=np.arctan2(h[0],h[2])
     a,b=project(original,r),project(box,r)
     if a is None or b is None:row['reasons'].append(arm+'_near_plane');break
     fs.append({'frame':i,'phase':phase,'dx':b['cx']-a['cx'],'visible_fraction':b['visible_fraction'],'area_fraction':b['area_fraction']})
     if b['visible_fraction']<.95 or b['area_fraction']<.025:row['reasons'].append(arm+'_projection_visibility')
    row['arms'][arm]={'frames':fs,'late_mean_dx':float(np.mean([f['dx'] for f in fs[4:]])) if len(fs)==8 else None}
   a=row['arms']['delayed']['late_mean_dx'];b=row['arms']['advanced']['late_mean_dx']
   if a is None or b is None:row['reasons'].append('incomplete_projection')
   elif min(abs(a),abs(b))<8 or a*b>=0:row['reasons'].append('insufficient_opposite_signal')
   else:
    for arm in ('delayed','advanced'):
     mean=row['arms'][arm]['late_mean_dx']
     if any(f['dx']*mean<=0 for f in row['arms'][arm]['frames'][4:]):row['reasons'].append(arm+'_late_sign_inconsistent')
  row['reasons']=sorted(set(row['reasons']));row['signal_support_pass']=not row['reasons'];rows.append(row)
  print('SOURCE_SIGNAL',n,amp,row['signal_support_pass'],{k:v['late_mean_dx'] for k,v in row['arms'].items()},row['reasons'],flush=True)
assert source.stat().st_size==protocol['source_size'] and source.stat().st_mtime_ns==protocol['source_mtime_ns']
save(OUT/'results.json',{'eligibility':eligibility,'support':support,'rows':rows,'source_only':True,'generation_run':False,'elapsed_seconds':time.monotonic()-start,'track_cache_sha256':sha(OUT/'extended_tracks.pkl')})
print('SOURCE_SIGNAL_PRECHECK_COMPLETE',time.monotonic()-start,flush=True)
