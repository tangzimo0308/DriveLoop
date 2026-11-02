"""Validate namespace-safe source copies and all fixed geometry points; no generation."""
import json,pickle,hashlib,time,lmdb
from pathlib import Path
from scripts.copy_val_native_v2_20261019 import copy_native_explicit
from scripts.geometry_with_source_support import check_geometry
ROOT=Path('/home/lesliton/projects/DriveLoop');OUT=ROOT/'outputs/revision_20261019_val_source_geometry_v2';IN=ROOT/'outputs/revision_20261019_val_source_projection';MEAS=ROOT/'outputs/revision_20261019_val_source_measurability';PLAN=ROOT/'outputs/revision_20261019_measurable_source_plan/protocol.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
OUT.mkdir(exist_ok=False);start=time.monotonic();plan=read(PLAN);s=read(IN/'summary.json');mres=read(MEAS/'results.json')
assert mres['complete'] and sha(MEAS/'protocol.json')==mres['protocol_sha256']
assert sha(PLAN)==read(IN/'protocol.json')['amended_plan_sha256']
assert sha(IN/'selected_candidates.pkl')==s['selected_pickle_sha256']
source=Path(plan['source_dataset']);assert sha(source/'labels/data.pkl')==read(IN/'protocol.json')['source_sha256']
pool=pickle.loads((IN/'selected_candidates.pkl').read_bytes());pool=[c for c in pool if c['candidate_index'] in mres['passed_primary_source_measurability']];assert [c['candidate_index'] for c in pool]==[26,568,629]
fixed={str(p):sha(p) for p in [PLAN,IN/'selected_candidates.pkl',IN/'summary.json',MEAS/'protocol.json',MEAS/'results.json',ROOT/'scripts/geometry_with_source_support.py',ROOT/'scripts/copy_val_native_v2_20261019.py',Path(__file__).resolve()]}
phases=[-.5,-.375,-.25,-.125,0,.125,.25,.375,.5]
protocol={'written_before_geometry':True,'generation_run':False,'source_dataset':str(source),'candidate_order':[c['candidate_index'] for c in pool],'namespace':'cam_all_val/v0.0.2; val_candidate prefix; scene+instance authoritative','phases_s':phases,'fixed_comparator_phases_s':[-.375,-.25,-.125,.125,.25,.375],'bounds_rule':'maximal contiguous passing grid component containing zero; must contain all fixed comparator phases','thresholds':{'clearance_m':.5,'road_fraction':.99,'visible_fraction':.95,'bottom_change_m':.15,'sweep_step_s':.02},'fixed_hashes':fixed,'no_generated_scores':True,'no_threshold_changes':True,'all_failures_retained':True}
save(OUT/'protocol.json',protocol)
original_records=pickle.loads((source/'labels/data.pkl').read_bytes());rows=[]
for c in pool:
 assert c['source_dataset']==str(source) and c['scene_token'] not in plan['excluded_scenes']
 # Check each serialized source label against the actual validation split at the same index.
 for startidx,group in zip(c['source_camera_starts'],c['groups']):
  for offset,a in enumerate(group):
   idx=startidx+offset;assert int(a['data_index'])==idx;b=original_records[idx];assert a.keys()==b.keys()
   for key in a:assert pickle.dumps(a[key],protocol=4)==pickle.dumps(b[key],protocol=4),(c['candidate_index'],idx,key)
 case=OUT/f"val_candidate{c['candidate_index']}";case.mkdir();m=copy_native_explicit(c,case/'original',protocol,source)
 assert m['source_dataset']==str(source) and m['source_candidate_id']==f"val_candidate{c['candidate_index']}" and m['binding']['ready']
 copied=pickle.loads((Path(m['dataset'])/'labels/data.pkl').read_bytes());assert len(copied)==144
 byte_counts={}
 for part in ('images','hdmaps'):
  src=lmdb.open(str(source/part),readonly=True,lock=False,readahead=False);dst=lmdb.open(str(Path(m['dataset'])/part),readonly=True,lock=False,readahead=False)
  with src.begin() as st,dst.begin() as dt:
   for local,old in enumerate(m['source_data_indices']):
    a=st.get(str(old).encode());b=dt.get(str(local).encode());assert a is not None and a==b and hashlib.sha256(a).hexdigest()==m['image_and_map_hashes'][part][local]
  src.close();dst.close();byte_counts[part]=len(copied)
 row={'candidate_index':c['candidate_index'],'source_split':c['source_split'],'scene_token':c['scene_token'],'instance_token':c['instance_token'],'source_copy_bytes_checked':byte_counts,'source_labels_checked':144,'original_manifest':str(case/'original/manifest.json'),'phases':[]}
 for i,phase in enumerate(phases):
  dest=case/f'phase_{i:02d}';gm=check_geometry(m,dest,phase,c['extended_track']);chk=read(dest/'geometry_checks.json')
  pr={'phase_s':phase,'passed':chk['passed'],'reasons':sorted({r for t in chk['trials'] for r in t['reasons']}),'check_path':str(dest/'geometry_checks.json'),'check_sha256':sha(dest/'geometry_checks.json')}
  if gm:
   mean=sum(f['projected_center_shift_x_px_at448'] for f in gm['geometry']['frames'][4:])/4
   expected=next(v['source_mean_dx_px'] for v in c['source_screen']['curve'] if v['phase_s']==phase);assert abs(mean-expected)<1e-3
   pr.update(manifest=str(dest/'manifest.json'),manifest_sha256=sha(dest/'manifest.json'),source_mean_dx_px=mean)
  row['phases'].append(pr);print('GEOMETRY',c['candidate_index'],phase,pr['passed'],pr['reasons'],flush=True)
  save(OUT/'partial_results.json',rows+[row])
 flags=[v['passed'] for v in row['phases']];zero=phases.index(0);component=[]
 if flags[zero]:
  lo=hi=zero
  while lo>0 and flags[lo-1]:lo-=1
  while hi<len(phases)-1 and flags[hi+1]:hi+=1
  component=phases[lo:hi+1]
 row.update(all_grid_passed=all(flags),passing_component_s=component,grid_bounds_s=[component[0],component[-1]] if component else None,eligible_for_runtime=set(protocol['fixed_comparator_phases_s']).issubset(component))
 rows.append(row)
for p,h in fixed.items():assert sha(p)==h,p
assert sha(source/'labels/data.pkl')==read(IN/'protocol.json')['source_sha256']
result={'complete':True,'generation_run':False,'actual_model_loader_calls':0,'rows':rows,'eligible_for_runtime':[r['candidate_index'] for r in rows if r['eligible_for_runtime']],'geometry_points_checked':len(rows)*len(phases),'elapsed_seconds':time.monotonic()-start,'protocol_sha256':sha(OUT/'protocol.json'),'limits':['Known road/obstacle checks do not prove physical safety.','Finite passing grid does not authorize intermediate parameters.','Actual DD2 input and cached validation text compatibility still unverified.']}
save(OUT/'results.json',result);print('VAL_GEOMETRY_COMPLETE',result['eligible_for_runtime'],result['elapsed_seconds'],flush=True)
