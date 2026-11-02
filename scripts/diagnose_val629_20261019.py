import json,pickle,hashlib
from pathlib import Path
import numpy as np
from dreamer_datasets import boxes3d_utils
ROOT=Path('/home/lesliton/projects/DriveLoop');CASE=ROOT/'outputs/revision_20261019_val_inputs/val_candidate629';OUT=ROOT/'outputs/revision_20261019_val629_input_diagnosis';OUT.mkdir(exist_ok=False)
m=json.loads((CASE/'manifest.json').read_text())['selected'][0];source=m['source'];old=pickle.loads((Path(source['dataset'])/'labels/data.pkl').read_bytes());newpath=Path(m['arms']['advanced']['edited_dataset'])/'labels/data.pkl';new=pickle.loads(newpath.read_bytes());rows=[]
for cam in range(6):
 for step in range(8):
  idx=cam*24+step*3;a,b=old[idx],new[idx]
  for j,token in enumerate(b['instance_tokens']):
   if token!=source['instance_token']:continue
   z0=boxes3d_utils.boxes3d_to_corners3d(a['boxes3d'][j:j+1],rot_axis=1)[0,:,2];z1=boxes3d_utils.boxes3d_to_corners3d(b['boxes3d'][j:j+1],rot_axis=1)[0,:,2]
   if float(z1.mean())<=0:rows.append({'record_index':idx,'camera':b['cam_type'],'frame':step,'instance_token':token,'label':b['ori_labels3d'][j],'original_mean_corner_depth_m':float(z0.mean()),'advanced_mean_corner_depth_m':float(z1.mean()),'original_min_corner_depth_m':float(z0.min()),'advanced_min_corner_depth_m':float(z1.min())})
assert rows and all(x['label']=='vehicle.bus.rigid' for x in rows)
r={'diagnosis':'Retained target bus has nonpositive mean corner depth in a sampled camera after advanced phase; DD2 transform requires positive mean corner depth before cropping.','source_candidate_id':source['source_candidate_id'],'scene_token':source['scene_token'],'violations':rows,'actual_model_loader_calls_before_failure':len(list(CASE.glob('audit_*/generation.json'))),'new_loader_calls':0,'generation_run':False,'original_labels_sha256':hashlib.sha256((Path(source['dataset'])/'labels/data.pkl').read_bytes()).hexdigest(),'advanced_labels_sha256':hashlib.sha256(newpath.read_bytes()).hexdigest(),'scope':'Source geometry passed tested front-view and map constraints; this additional multiview runtime requirement failed. No thresholds or source records changed.'}
(OUT/'results.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
