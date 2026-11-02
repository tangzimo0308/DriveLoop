"""Only source-projection candidates proceed to original geometry guards."""
import json,pickle,time
from pathlib import Path
from scripts.geometry_with_source_support import check_geometry,sha,save
ROOT=Path('/home/lesliton/projects/DriveLoop');S=ROOT/'outputs/revision_20261008_signal_precheck';OUT=S/'geometry';OUT.mkdir(exist_ok=False);start=time.monotonic()
r=json.loads((S/'results.json').read_text());tracks=pickle.loads((S/'extended_tracks.pkl').read_bytes());assert sha(S/'extended_tracks.pkl')==r['track_cache_sha256']
passed=[x for x in r['rows'] if x['signal_support_pass']];assert {x['candidate_index'] for x in passed}=={1583}
source=ROOT/'outputs/revision_20261007_native_multiscene_source/candidate1583';m=json.loads((source/'original/manifest.json').read_text());track=tracks[1583]
protocol={'source_screen_results_sha256':sha(S/'results.json'),'extended_tracks_sha256':sha(S/'extended_tracks.pkl'),'helper_sha256':sha(ROOT/'scripts/geometry_with_source_support.py'),'driver_sha256':sha(__file__),'candidate_index':1583,'amplitude_order':[.25,.5],'stop_at_first_passing_amplitude':True,'geometry_rules_unchanged':True,'source_only':True,'no_dynamics_validation':True,'no_generation':True}
save(OUT/'protocol.json',protocol)
# Regression: extending support must not change an already-supported edit.
reg=check_geometry(m,OUT/'regression_delayed0125',.125,track)
assert reg is not None
old=json.loads((source/'geometry_delayed/manifest.json').read_text())
assert reg['edited_labels_sha256']==old['edited_labels_sha256'],'extended track changed previously supported poses'
save(OUT/'regression.json',{'passed':True,'unchanged_prior_delayed0125_labels_sha256':reg['edited_labels_sha256']});print('REGRESSION_EXACT_PASS',flush=True)
rows=[]
for signal in passed:
 amp=signal['amplitude_seconds'];record={'candidate_index':1583,'amplitude':amp,'arms':{}}
 for arm,delay in [('delayed',amp),('advanced',-amp)]:
  dest=OUT/(arm+str(amp).replace('.',''));v=check_geometry(m,dest,delay,track);g=json.loads((dest/'geometry_checks.json').read_text())
  record['arms'][arm]={'passed':v is not None,'directory':str(dest),'geometry':g}
 record['passed']=all(v['passed'] for v in record['arms'].values());rows.append(record)
 save(OUT/'results.json',{'complete':False,'rows':rows});print('GEOMETRY',amp,record['passed'],{k:v['geometry']['trials'][-1]['reasons'] for k,v in record['arms'].items()},flush=True)
 if record['passed']:break
save(OUT/'results.json',{'complete':True,'rows':rows,'elapsed_seconds':time.monotonic()-start,'source_only':True,'generation_run':False})
print('SOURCE_GEOMETRY_PRECHECK_COMPLETE',time.monotonic()-start,flush=True)
