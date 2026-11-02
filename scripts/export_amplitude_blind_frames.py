import json,hashlib,shutil
from pathlib import Path
ROOT=Path('/home/lesliton/projects/DriveLoop');RUN=ROOT/'outputs/dynamic_amplitude_20261014_222304_43184';OUT=ROOT/'outputs/revision_20261015_blind_frames';OUT.mkdir(exist_ok=False)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=read(RUN/'summary.json');ev=read(ROOT/'outputs/revision_20261015_amplitude_yolo/protocol.json')
assert sha(RUN/'summary.json')==ev['source_hashes'][str(RUN/'summary.json')]
rows={};selected=[];inputs={}
for m in s['methods']:
 rp=RUN/m['direction']/m['method']/'result.json';assert sha(rp)==ev['source_hashes'][str(rp)];inputs[str(rp)]=sha(rp)
 result=read(rp);i=m['best_measured']['iteration'];assert i==result['best_generation']['iteration']==ev['fixed_selected_indices'][m['direction']+'/'+m['method']]
 f=result['attempt_history'][i]['evaluation']['metrics']['native_phase_feedback'];pr=f['provenance']
 group={'direction':m['direction'],'method':m['method'],'selected_iteration':i,'target_mean_dx_px':f['contract']['target_dx_px'],'current':[],'reference':[]}
 for role in ['current','reference']:
  c=pr[role]
  for j in range(4,8):
   frame=c['frames'][j];h=frame['sha256'];assert sha(frame['path'])==h
   group[role].append(h)
   identity={'role':role,'seed':c['seed_offset'],'arm':c['arm'],'frame':j,'scene_token':c['scene_token'],'instance_token':c['instance_token']}
   if h not in rows:
    name=h+'.png';shutil.copyfile(frame['path'],OUT/name);rows[h]={'filename':name,'sha256':h,'source_path':frame['path'],'identities':[]}
   if identity not in rows[h]['identities']:rows[h]['identities'].append(identity)
 selected.append(group)
assert len(rows)==32
manifest={'run':str(RUN),'summary_sha256':sha(RUN/'summary.json'),'input_hashes':inputs,'selection':'All six frozen selected outputs and matching same-seed original references; all four late frames; exact duplicates deduplicated','images':list(rows.values()),'selected_groups':selected,'width':448,'height':256,'generation_run':False,'detector_inference_run':False}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('EXPORTED',len(rows),OUT)
