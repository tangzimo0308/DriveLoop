"""Copy exactly the admitted source window, preserving encoded image/map bytes."""
import copy,hashlib,json,pickle
from pathlib import Path
import lmdb
from driveloop.source_sample_binding import build_source_sample_binding
R=Path.cwd();O=R/'outputs/revision_20261006_source_screen_expanded';screen=json.loads((O/'geometry_screen.json').read_text());selected=[r for r in screen['windows'] if r['status']=='geometry_admitted'];assert len(selected)==1
c=pickle.loads((O/'selected_window.pkl').read_bytes());assert c['candidate_index']==selected[0]['candidate_index'];dest=O/f"candidate{c['candidate_index']}";dest.mkdir(exist_ok=False);dataset=dest/'dataset';dataset.mkdir()
source=Path('/data/projects/DriveLoop/data/processed/nuscenes/cam_all_train/v0.0.2');names=['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_RIGHT','CAM_BACK','CAM_BACK_LEFT'];old_to_new={old:i*24 for i,old in enumerate(c['source_camera_starts'])};records=[];source_data_indices=[]
for group in c['groups']:
 for r in group:
  new=copy.deepcopy(r);source_data_indices.append(int(new['data_index']));new['data_index']=len(records);new['multiview_start_idx']={k:old_to_new[v] for k,v in new.get('multiview_start_idx',{}).items() if v in old_to_new};records.append(new)
(dataset/'labels').mkdir();(dataset/'labels/data.pkl').write_bytes(pickle.dumps(records,protocol=4));config=json.loads((source/'labels/config.json').read_text());config['data_size']=len(records);(dataset/'labels/config.json').write_text(json.dumps(config,indent=2)+'\n')
hashes={}
for component in ('images','hdmaps'):
 (dataset/component).mkdir();src=lmdb.open(str(source/component),readonly=True,lock=False,readahead=False);dst=lmdb.open(str(dataset/component),map_size=2**30);hashes[component]=[]
 with src.begin() as read,dst.begin(write=True) as write:
  for i,oldidx in enumerate(source_data_indices):
   b=read.get(str(oldidx).encode());assert b is not None,(component,oldidx);write.put(str(i).encode(),b);hashes[component].append(hashlib.sha256(b).hexdigest())
 src.close();dst.close();conf=json.loads((source/component/'config.json').read_text());conf['data_size']=len(records);conf.pop('data_path',None);(dataset/component/'config.json').write_text(json.dumps(conf,indent=2)+'\n')
(dataset/'config.json').write_text(json.dumps({'_class_name':'Dataset','config_paths':['labels/config.json','images/config.json','hdmaps/config.json']},indent=2)+'\n')
identity={'candidate':f"candidate{c['candidate_index']}",'provenance':'score-blind source geometry screening; new diagnostic source, not historical reproduction','frame_summaries':[{k:group[i*3][k] for k in ('cam_type','sample_token','scene_token','frame_idx','cam_token','cam_timestamp')} for group in c['groups'] for i in range(8)]};identity_path=dest/'identity.json';identity_path.write_text(json.dumps(identity,indent=2)+'\n')
binding=build_source_sample_binding(dataset,source_candidate_id=identity['candidate'],instance_token=c['instance_token'],scene_token=c['scene_token'],identity_summary_path=identity_path,frame_num=8,hz_factor=3,video_split_rate=1,multiview=True)
assert binding['ready'],binding
manifest={'candidate_index':c['candidate_index'],'source_candidate_id':identity['candidate'],'category':'bus','prompt':'urban street, a bus cuts in from the left toward the ego vehicle','dataset':str(dataset),'identity':str(identity_path),'scene_token':c['scene_token'],'instance_token':c['instance_token'],'source_dataset':str(source),'source_record_indices':c['source_camera_starts'],'image_and_map_hashes':hashes,'source_data_indices':source_data_indices,'geometry_admission':selected[0],'binding':binding,'no_generation_yet':True}
(dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('SUBSET_READY',dest, 'records',len(records),flush=True)
