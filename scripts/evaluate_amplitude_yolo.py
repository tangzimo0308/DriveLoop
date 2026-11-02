"""Secondary evaluation of frozen controller selections; no regeneration or reselection."""
import json,time,hashlib,fcntl,subprocess,statistics
from pathlib import Path
from PIL import Image
import torch,ultralytics
from ultralytics import YOLO
from driveloop.native_phase_png import associate
ROOT=Path('/home/lesliton/projects/DriveLoop')
RUN=ROOT/'outputs/dynamic_amplitude_20261014_222304_43184'
OUT=ROOT/'outputs/revision_20261015_amplitude_yolo'
WEIGHT=ROOT/'yolov8x.pt'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def center(b):return (b[0]+b[2])/2
lock=open('/tmp/driveloop-m2-motion-guard.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
assert not gpu.stdout.strip(),'GPU occupied; do not compete with generation'
OUT.mkdir(exist_ok=False);start=time.monotonic()
s=read(RUN/'summary.json');audit=read(ROOT/'outputs/revision_20261014_amplitude_results_audit/results.json')
assert audit['passed'] and audit['summary_sha256']==sha(RUN/'summary.json')
assert s['complete'] and len(s['methods'])==6
methods=[];images={};source_hashes={str(RUN/'summary.json'):sha(RUN/'summary.json')}
for m in s['methods']:
 p=RUN/m['direction']/m['method']/'result.json';result=read(p);source_hashes[str(p)]=sha(p)
 assert result['best_generation']['iteration']==m['best_measured']['iteration']
 attempts=result['attempt_history'];assert len(attempts)==3
 for a in attempts:
  pr=a['evaluation']['metrics']['native_phase_feedback']['provenance']
  for role in ['reference','current']:
   clip=pr[role];assert sha(clip['evidence_path'])==clip['evidence_sha256']
   for f in clip['frames']:
    assert sha(f['path'])==f['sha256']
    with Image.open(f['path']) as im:assert im.size==(448,256) and im.mode=='RGB'
    images.setdefault(f['sha256'],f['path'])
 methods.append((m,attempts))
protocol={'stage':'secondary_development_evaluation','written_before_yolo_inference':True,
 'run':str(RUN),'script_sha256':sha(__file__),'weight_path':str(WEIGHT),'weight_sha256':sha(WEIGHT),
 'ultralytics_version':ultralytics.__version__,'torch_version':torch.__version__,
 'prediction':{'conf':.5,'iou':.7,'imgsz':640,'device':0,'half':False,'augment':False,'max_det':300,'rect':True},
 'association':'bus, confidence >= 0.5, projected target IoU >= 0.3, exactly one; missing/ambiguous remains missing',
 'main_selection':'Evaluate the frozen Faster R-CNN selected candidate per method/direction. Never reselect by YOLO.',
 'all_attempts':'Report every candidate as secondary diagnostic, no oracle minimum',
 'metric':'Absolute error of mean matched center displacement on frames 4-7; same-seed original reference. Four pairs required.',
 'additional_metric':'Mean per-frame absolute displacement error versus the INITIAL target geometry, never the refined conditioning geometry.',
 'fixed_selected_indices':{m['direction']+'/'+m['method']:m['best_measured']['iteration'] for m,_ in methods},
 'source_hashes':source_hashes,'unique_png_hashes':sorted(images),
 'limits':['YOLO did not drive this amplitude controller, but was used in earlier project development; not a wholly untouched evaluator.',
 'Post hoc analysis of the same inspected scene, not independent held-out data.',
 'Equal numeric detector confidence thresholds are not calibrated equal operating points.',
 'Condition-guided association and both detectors remain proxy measurements, not physical ground truth.']}
save(OUT/'protocol.json',protocol)
torch.set_num_threads(4);model=YOLO(str(WEIGHT));raw={}
for k,(h,path) in enumerate(sorted(images.items())):
 pred=model.predict(source=path,verbose=False,save=False,**protocol['prediction'])[0]
 raw[h]=[{'label':str(pred.names[int(cl)]),'score':float(sc),'box':[float(v) for v in box]} for box,sc,cl in zip(pred.boxes.xyxy.cpu().tolist(),pred.boxes.conf.cpu().tolist(),pred.boxes.cls.cpu().tolist())]
 if (k+1)%32==0:print('INDEPENDENT_PNG',k+1,'/',len(images),flush=True)
save(OUT/'raw_detections.json',raw)
rows=[]
for m,attempts in methods:
 base=attempts[0]['evaluation']['metrics']['native_phase_feedback']['provenance']
 target_dx=[center(base['current']['frames'][j]['projected_box'])-center(base['reference']['frames'][j]['projected_box']) for j in range(8)]
 target=attempts[0]['generation']['metadata']['native_phase_contract']['target_dx_px']
 assert abs(statistics.mean(target_dx[4:])-target)<1e-3
 derived=[]
 for i,a in enumerate(attempts):
  f=a['evaluation']['metrics']['native_phase_feedback'];pr=f['provenance'];assoc={}
  for role in ['reference','current']:
   assoc[role]=[{'frame':j,**associate(raw[fr['sha256']],fr['projected_box'])} for j,fr in enumerate(pr[role]['frames'])]
  dx=[center(assoc['current'][j]['target']['box'])-center(assoc['reference'][j]['target']['box']) if assoc['current'][j]['target'] is not None and assoc['reference'][j]['target'] is not None else None for j in range(8)]
  measurable=all(x is not None for x in dx[4:]);observed=statistics.mean(dx[4:]) if measurable else None
  old_dx=f['late_deltas'];old_mae=statistics.mean(abs(v-t) for v,t in zip(old_dx,target_dx[4:]))
  derived.append({'iteration':i,'phase_s':f['contract']['phase_s'],'is_frozen_selected':i==m['best_measured']['iteration'],
   'association':assoc,'frame_deltas':dx,'measurable':measurable,'mean_dx_px':observed,
   'absolute_mean_error_px':abs(target-observed) if measurable else None,
   'per_frame_mae_px':statistics.mean(abs(v-t) for v,t in zip(dx[4:],target_dx[4:])) if measurable else None,
   'controller_absolute_mean_error_px':abs(f['error_px']),'controller_per_frame_mae_px':old_mae})
 rows.append({'direction':m['direction'],'method':m['method'],'target_mean_dx_px':target,'initial_target_frame_dx_px':target_dx,'frozen_selected_iteration':m['best_measured']['iteration'],'selected':derived[m['best_measured']['iteration']],'attempts':derived})
comparisons=[]
for direction in ['delayed','advanced']:
 d=next(x for x in rows if x['direction']==direction and x['method']=='direct_target')['selected']
 f=next(x for x in rows if x['direction']==direction and x['method']=='residual_feedback')['selected']
 comparisons.append({'direction':direction,'both_measurable':d['measurable'] and f['measurable'],'feedback_minus_direct_error_px':f['absolute_mean_error_px']-d['absolute_mean_error_px'] if d['measurable'] and f['measurable'] else None})
for p,h in source_hashes.items():assert sha(p)==h
assert sha(WEIGHT)==protocol['weight_sha256']
result={'complete':True,'generation_run':False,'detector_inference_run':True,'unique_png_inferences':len(images),'candidate_attempts':18,'measurable_attempts':sum(a['measurable'] for r in rows for a in r['attempts']),'methods':rows,'comparisons':comparisons,'elapsed_seconds':time.monotonic()-start,'protocol_sha256':sha(OUT/'protocol.json'),'raw_detection_sha256':sha(OUT/'raw_detections.json'),'limits':protocol['limits']}
save(OUT/'results.json',result)
for r in rows:print('SELECTED',r['direction'],r['method'],r['frozen_selected_iteration'],r['selected']['absolute_mean_error_px'],r['selected']['per_frame_mae_px'])
print('COMPARISONS',comparisons)
print('INDEPENDENT_EVALUATION_COMPLETE',len(images),result['measurable_attempts'],result['elapsed_seconds'],OUT,flush=True)
