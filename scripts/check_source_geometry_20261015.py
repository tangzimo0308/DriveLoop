"""Check every predeclared phase for both frozen source candidates, no generation."""
import json,pickle,hashlib,time
from pathlib import Path
from scripts.geometry_with_source_support import copy_native,check_geometry
ROOT=Path('/home/lesliton/projects/DriveLoop');IN=ROOT/'outputs/revision_20261015_source_expansion';OUT=ROOT/'outputs/revision_20261015_source_geometry';OUT.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
start=time.monotonic();s=read(IN/'summary.json');protocol=read(IN/'protocol.json');assert s['complete_source_inventory'] and sha(IN/'selected_candidates.pkl')==s['selected_pickle_sha256']
pool=pickle.loads((IN/'selected_candidates.pkl').read_bytes());assert len(pool)==2
plan={'created_before_geometry':True,'candidate_order':[c['candidate_index'] for c in pool],'source_inventory_sha256':sha(IN/'summary.json'),'source_selection_protocol_sha256':sha(IN/'protocol.json'),'selected_pickle_sha256':sha(IN/'selected_candidates.pkl'),'geometry_helper_sha256':sha(ROOT/'scripts/geometry_with_source_support.py'),'script_sha256':sha(Path(__file__)),'phases_s':protocol['source_curve_grid_s'],'checks':{'known_obstacle_clearance_m':.5,'road_fraction':.99,'visible_fraction':.95,'max_bottom_change_m':.15,'sweep_max_dt_s':.02},'all_candidates_all_phases_reported':True,'no_retry_or_threshold_change':True,'generation_run':False,'limits':['Finite source grid does not admit intermediate parameters; fresh exact-parameter checks remain mandatory.','Known annotations and road map checks are not full physical dynamics or safety validation.']}
save(OUT/'protocol.json',plan);rows=[]
for c in pool:
 case=OUT/f"candidate{c['candidate_index']}";case.mkdir()
 m=copy_native(c,case/'original',plan);r={'candidate_index':c['candidate_index'],'scene_token':c['scene_token'],'instance_token':c['instance_token'],'original_manifest':str(case/'original/manifest.json'),'phases':[]}
 for i,phase in enumerate(plan['phases_s']):
  dest=case/f'phase_{i:02d}';gm=check_geometry(m,dest,phase,c['extended_track']);checks=read(dest/'geometry_checks.json')
  phase_row={'phase_s':phase,'passed':checks['passed'],'check_path':str(dest/'geometry_checks.json'),'check_sha256':sha(dest/'geometry_checks.json'),'reasons':sorted({reason for tr in checks['trials'] for reason in tr['reasons']})}
  if gm is not None:
   mean=sum(f['projected_center_shift_x_px_at448'] for f in gm['geometry']['frames'][4:])/4
   expected=next(x for x in c['source_screen']['curve'] if x['phase_s']==phase)['source_mean_dx_px'];assert abs(mean-expected)<1e-3
   phase_row.update(manifest=str(dest/'manifest.json'),manifest_sha256=sha(dest/'manifest.json'),source_mean_dx_px=mean,min_sample_clearance_m=min(x['clearance_m'] for x in gm['geometry']['frames']),min_interval_clearance_m=min(x['clearance_m'] for x in gm['geometry']['intervals']))
  r['phases'].append(phase_row);print('GEOMETRY',c['candidate_index'],phase,phase_row['passed'],phase_row['reasons'],flush=True)
  save(OUT/'partial_results.json',rows+[r])
 r['all_grid_phases_passed']=all(x['passed'] for x in r['phases']);rows.append(r)
assert sha(ROOT/'scripts/geometry_with_source_support.py')==plan['geometry_helper_sha256']
res={'complete':True,'candidates':rows,'all_grid_passed_candidates':[r['candidate_index'] for r in rows if r['all_grid_phases_passed']],'runtime_input_verified':False,'generation_run':False,'elapsed_seconds':time.monotonic()-start,'protocol_sha256':sha(OUT/'protocol.json'),'next_stage':'Scene-specific DD2 runtime input admission before any generation'}
save(OUT/'results.json',res);print('SOURCE_GEOMETRY_COMPLETE',res['all_grid_passed_candidates'],res['elapsed_seconds'],flush=True)
