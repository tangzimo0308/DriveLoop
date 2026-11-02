"""Bounded, score-blind source shortlist; no model or image loading."""
import json,pickle,time,hashlib,copy
from pathlib import Path
from collections import Counter
from scripts.run_dd2_batch_sampler_audit import candidate_camera_starts
from scripts.find_candidate_actor import instances_across_frames
root=Path.cwd();out=root/'outputs/revision_20261006_source_screen';out.mkdir(exist_ok=True)
source=Path('/data/projects/DriveLoop/data/processed/nuscenes/cam_all_train/v0.0.2');path=source/'labels/data.pkl'
protocol={'task':'source_window_screen_after_candidate28_geometry_failure','source_dataset':str(source),'frame_num':8,'hz_factor':3,'video_split_rate':1,'old_scene_excluded':'6f5133fe62b240e797bac25aeff8b531','selection':'ascending source candidate index; up to two eligible windows per scene; first 40 shortlisted; identity must occur in all 8 front frames; no generated-video score','candidate_cap':40,'per_scene_cap':2,'geometry_rules':{'source_and_target_endpoint_fraction':.95,'known_obstacle_clearance_m':.5,'all_six_views_all_annotated_classes':True},'claim':'exploratory source admission, not an unbiased evaluation sample; preserve failures and report screened denominator'}
(out/'selection_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
t0=time.monotonic();print('LOADING_LABELS',flush=True)
records=pickle.loads(path.read_bytes());starts=candidate_camera_starts(records,frame_num=8,hz_factor=3,video_split_rate=1,multiview=True);print('LOADED',len(records),'starts',len(starts),'seconds',time.monotonic()-t0,flush=True)
scene_count=Counter();counts=Counter();chosen=[];ledger=[]
for index,group in enumerate(starts):
 front=[records[group[1]+i*3] for i in range(8)];scene=front[0]['scene_token'];entry={'candidate_index':index,'scene_token':scene}
 if scene==protocol['old_scene_excluded']:reason='old_failed_scene'
 elif scene_count[scene]>=2:reason='scene_shortlist_cap'
 else:
  presence,cats=instances_across_frames(front,'vehicle.bus');common=sorted(k for k,v in presence.items() if len(v)==8)
  if len({r['scene_token'] for r in front})!=1:reason='scene_changes'
  elif not common:reason='no_bus_instance_continuous_8_frames'
  else:
   reason='shortlisted';scene_count[scene]+=1
   groups=[[copy.deepcopy(records[start+j]) for j in range(24)] for start in group]
   chosen.append({'candidate_index':index,'source_camera_starts':group,'scene_token':scene,'scene_description':front[0]['scene_description'],'instance_token':common[0],'eligible_instances':common,'groups':groups,'front_frame_indices':[r['frame_idx'] for r in front]})
 counts[reason]+=1;entry['status']=reason;ledger.append(entry)
 if len(chosen)>=40:break
(out/'shortlist.pkl').write_bytes(pickle.dumps(chosen,protocol=4))
report={'source_records':len(records),'source_candidates':len(starts),'visited_candidates':len(ledger),'shortlisted':len(chosen),'scenes':len(scene_count),'counts':dict(counts),'seconds':time.monotonic()-t0,'source_size_bytes':path.stat().st_size,'source_mtime_ns':path.stat().st_mtime_ns,'ledger':ledger,'shortlist':[{k:v for k,v in c.items() if k!='groups'} for c in chosen]}
(out/'shortlist.json').write_text(json.dumps(report,indent=2)+'\n');print('SHORTLIST_READY',json.dumps({k:report[k] for k in ('shortlisted','scenes','visited_candidates','seconds')}),flush=True)
