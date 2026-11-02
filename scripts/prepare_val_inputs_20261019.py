import json,pickle,hashlib
from pathlib import Path
ROOT=Path('/home/lesliton/projects/DriveLoop');G=ROOT/'outputs/revision_20261019_val_source_geometry_v2';OUT=ROOT/'outputs/revision_20261019_val_inputs'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
g=read(G/'results.json');assert g['complete'] and g['eligible_for_runtime']==[568,629]
assert sha(G/'protocol.json')==g['protocol_sha256']
for p,h in read(G/'protocol.json')['fixed_hashes'].items():assert sha(p)==h,p
OUT.mkdir(exist_ok=False)
for row in g['rows']:
 if not row['eligible_for_runtime']:continue
 n=row['candidate_index'];source=read(row['original_manifest']);case=OUT/f'val_candidate{n}';case.mkdir()
 (case/'original').symlink_to(Path(row['original_manifest']).parent,target_is_directory=True)
 arms={}
 for name,phase in [('original',0),('delayed',.25),('advanced',-.25)]:
  pr=next(x for x in row['phases'] if x['phase_s']==phase);assert pr['passed'] and sha(pr['manifest'])==pr['manifest_sha256'];arms[name]=read(pr['manifest'])
 protocol={'stage':'validation_split_initial_three_actual_inputs','source_split':'cam_all_val/v0.0.2','candidate_index':n,'source_candidate_id':source['source_candidate_id'],'scene_token':source['scene_token'],'instance_token':source['instance_token'],'arms':{'original':0,'delayed':.25,'advanced':-.25},'source_grid_bounds_s':row['grid_bounds_s'],'geometry_results_sha256':sha(G/'results.json'),'expected_actual_loader_calls':3,'diffusion_calls':0,'generation_run':False,'scope':'Original and initial +/-0.25 only; other parameters require exact admission before generation.','failure_policy':'Stop and retain evidence; no text-cache fallback, no source rebinding, no retries.'}
 save(case/'protocol.json',protocol);save(case/'selection_ledger.json',g)
 save(case/'manifest.json',{'selected_count':1,'selected':[{'source':source,'arms':arms,'case':str(case)}],'complete':True,'generation_run':False,'protocol':protocol})
 files=[case/'protocol.json',case/'selection_ledger.json',case/'manifest.json',G/'protocol.json',G/'results.json',ROOT/'scripts/preflight_val_inputs_20261019.py',Path(__file__).resolve()]
 files.extend(Path(p) for p in read(G/'protocol.json')['fixed_hashes'])
 original=Path(row['original_manifest']).parent
 for sub in (original,Path(row['phases'][2]['manifest']).parent,Path(row['phases'][6]['manifest']).parent,Path(row['phases'][4]['manifest']).parent):
  files.extend(sub.rglob('*.json'));files.extend(sub.rglob('*.pkl'))
 files.extend(Path(source['dataset'])/part/'data.mdb' for part in ('images','hdmaps'))
 save(case/'launch_gate.json',{'fixed_hashes':{str(p):sha(p) for p in sorted(set(files))},'generation_run':False})
 print('VAL_INPUT_BUNDLE_READY',n,len(set(files)),flush=True)
