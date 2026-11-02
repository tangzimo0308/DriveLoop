"""Uniform source-only control range. No generated-output files read or model calls."""
import json,pickle,time
from pathlib import Path
from statistics import mean
from scripts.geometry_with_source_support import check_geometry,sha,save
from driveloop.phase_amplitude import propose_phase
ROOT=Path('/home/lesliton/projects/DriveLoop')
OUT=ROOT/'outputs/revision_20261013_phase_range'
S=ROOT/'outputs/revision_20261008_signal_precheck'
BASE=ROOT/'outputs/revision_20261007_native_multiscene_source/candidate1583/original'

def main():
 OUT.mkdir(exist_ok=False);start=time.monotonic()
 manifest=BASE/'manifest.json';trackfile=S/'extended_tracks.pkl';old=json.loads((S/'results.json').read_text())
 assert sha(trackfile)==old['track_cache_sha256'];tracks=pickle.loads(trackfile.read_bytes());m=json.loads(manifest.read_text())
 protocol={'source_only':True,'generation_run':False,'generated_results_read':False,'candidate_index':1583,
   'phase_seconds':[i/8 for i in range(-4,5)],'grid_basis':'Uniform dyadic grid within previously considered +/-0.5s bound; no fitting to detector error.',
   'max_proposal_step_s':.125,'software_deadband_px':1,'rule':'desired source shift = source shift at current phase + measured signed residual; bounded piecewise linear source inverse',
   'synthetic_probe':{'current_phase_s':0,'target_dx_px':4,'late_deltas':[0,0,0,0]},
   'proposal_validation':'One synthetic interior inverse proposal receives exact-parameter geometric checks; no generated residual used.',
   'geometry_guards_unchanged':{'clearance_m':.5,'road_fraction':.99,'visible_fraction':.95,'bottom_change_m':.15,'interval_step_s':.02},
   'input_sha256':{str(manifest):sha(manifest),str(trackfile):sha(trackfile),str(BASE/'dataset/labels/data.pkl'):sha(BASE/'dataset/labels/data.pkl'),str(ROOT/'scripts/geometry_with_source_support.py'):sha(ROOT/'scripts/geometry_with_source_support.py')},
   'code_sha256':{str(Path(__file__).resolve()):sha(__file__),str(ROOT/'driveloop/phase_amplitude.py'):sha(ROOT/'driveloop/phase_amplitude.py')},
   'limits':'Pointwise sampled/interval geometric constraints only, not dynamic safety or continuous-range certification. DD2 inputs not yet audited; all proposals forbidden for generation until separately validated.'}
 save(OUT/'protocol.json',protocol);rows=[]
 for i,phase in enumerate(protocol['phase_seconds']):
  dest=OUT/f'grid_{i:02d}';edited=check_geometry(m,dest,phase,tracks[1583]);g=json.loads((dest/'geometry_checks.json').read_text());trial=g['trials'][-1]
  frames=trial.get('frames',[])
  row={'phase_s':phase,'geometry_passed':bool(g['passed']),'source_mean_dx_px':mean(f['projected_center_shift_x_px_at448'] for f in frames[4:8]) if len(frames)==8 else None,
    'reasons':trial.get('reasons',[]),'directory':str(dest),'elapsed_total_seconds':time.monotonic()-start}
  if frames:
   row.update(min_sampled_clearance_m=min(f['clearance_m'] for f in frames),min_road_fraction=min(f['road_fraction'] for f in frames),min_visible_fraction=min(f['visible_fraction'] for f in frames),max_absolute_bottom_change_m=max(abs(f['bottom_change_m']) for f in frames))
  if trial.get('intervals'):row['min_interval_clearance_m']=min(f['clearance_m'] for f in trial['intervals'])
  if edited:row['edited_labels_sha256']=edited['edited_labels_sha256']
  rows.append(row);save(OUT/'partial_results.json',rows);print('SOURCE_RANGE_POINT',phase,row['geometry_passed'],row['source_mean_dx_px'],row['reasons'],round(time.monotonic()-start,2),flush=True)
 # Exact regression against the two previously approved +/-0.25s recipes.
 for phase,name in [(.25,'delayed025'),(-.25,'advanced025')]:
  oldm=json.loads((S/'geometry'/name/'manifest.json').read_text());row=next(r for r in rows if r['phase_s']==phase)
  assert row.get('edited_labels_sha256')==oldm['edited_labels_sha256']
 probe=None
 if all(r['geometry_passed'] for r in rows):
  probe=propose_phase(rows,0,4,[0]*4)
  path=OUT/'synthetic_inverse_probe';edited=check_geometry(m,path,probe['proposed_phase_s'],tracks[1583])
  g=json.loads((path/'geometry_checks.json').read_text());probe['exact_geometry_passed']=g['passed'];probe['geometry_reasons']=g['trials'][-1].get('reasons',[])
  if edited:probe['exact_source_mean_dx_px']=mean(f['projected_center_shift_x_px_at448'] for f in edited['geometry']['frames'][4:8])
  print('SYNTHETIC_INVERSE_PROBE',json.dumps(probe),flush=True)
 for p,h in protocol['input_sha256'].items():assert sha(p)==h,p
 save(OUT/'results.json',{'complete':True,'source_only':True,'generation_run':False,'generated_results_read':False,'rows':rows,'old_recipe_regression_passed':True,'synthetic_probe':probe,'elapsed_seconds':time.monotonic()-start})
 print('SOURCE_PHASE_RANGE_AUDIT_COMPLETE',OUT,flush=True)
if __name__=='__main__':main()
