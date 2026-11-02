import copy,json,pickle,hashlib,lmdb
from pathlib import Path
from driveloop.source_sample_binding import build_source_sample_binding
from scripts.run_bus_height_pair import save

def copy_native_explicit(selected,OUT,protocol,source):
 OUT.mkdir(parents=True,exist_ok=False)
 c=selected;(OUT/'selected_window.pkl').write_bytes(pickle.dumps(c,protocol=4))
 source=Path(source);assert source.resolve()==Path(c['source_dataset']).resolve() and source.resolve().name=='v0.0.2' and source.resolve().parent.name=='cam_all_val';dataset=OUT/'dataset';dataset.mkdir()
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
 identity={'candidate':f"val_candidate{c['candidate_index']}",'provenance':'native rigid-bus source visibility control',
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
