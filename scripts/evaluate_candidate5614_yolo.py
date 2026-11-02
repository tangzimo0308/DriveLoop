"""Fixed secondary detector diagnostic of all executed first candidates, no selection."""
import json,time,hashlib,fcntl,subprocess,statistics
from pathlib import Path
from PIL import Image
import torch,ultralytics
from ultralytics import YOLO
from driveloop.native_phase_png import associate
ROOT=Path('/home/lesliton/projects/DriveLoop');RUN=ROOT/'outputs/candidate5614_pilot_20261019_011254_9302';OUT=ROOT/'outputs/revision_20261019_candidate5614_yolo';WEIGHT=ROOT/'yolov8x.pt'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def center(b):return (b[0]+b[2])/2
lock=open('/tmp/driveloop-m2-motion-guard.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
OUT.mkdir(exist_ok=False);start=time.monotonic()
s=read(RUN/'summary.json');audit=read(ROOT/'outputs/revision_20261019_candidate5614_audit/results.json')
assert audit['passed'] and audit['summary_sha256']==sha(RUN/'summary.json')
prior=read(ROOT/'outputs/revision_20261015_amplitude_yolo/protocol.json')
assert sha(WEIGHT)==prior['weight_sha256']=='3df4ada6b4dad6d657868f2fdf7faecfb34dcfccf3a25c4b82079064718524c8'
methods=[];images={};clips={};source_hashes={str(RUN/'summary.json'):sha(RUN/'summary.json')}
for m in s['methods']:
 p=RUN/m['direction']/m['method']/'result.json';result=read(p);source_hashes[str(p)]=sha(p)
 assert m['best_measured'] is None and len(result['attempt_history'])==1
 a=result['attempt_history'][0];f=a['evaluation']['metrics']['native_phase_feedback'];pr=f['provenance']
 for role in ['reference','current']:
  clip=pr[role];assert sha(clip['evidence_path'])==clip['evidence_sha256']
  for fr in clip['frames']:
   assert sha(fr['path'])==fr['sha256']
   with Image.open(fr['path']) as im:assert im.size==(448,256) and im.mode=='RGB'
   images.setdefault(fr['sha256'],fr['path'])
  key='reference' if role=='reference' else m['direction']
  sig=[x['sha256'] for x in clip['frames']]
  if key in clips:assert sig==[x['sha256'] for x in clips[key]['frames']]
  else:clips[key]=clip
 methods.append((m,f))
assert len(images)==24 and set(clips)=={'reference','delayed','advanced'}
protocol={'written_before_inference':True,'source_run':str(RUN),'script_sha256':sha(__file__),'weight_path':str(WEIGHT),'weight_sha256':sha(WEIGHT),'prediction':prior['prediction'],'ultralytics_version':ultralytics.__version__,'torch_version':torch.__version__,'association':prior['association'],'selection':'None. All six first attempts are unmeasured under the controller; evaluate all without substitution or reselection.','scope':'Seed0 reference plus delayed and advanced first candidates; other methods have byte-identical frames. Seeds1/2 references are unused by all executed comparisons and excluded.','metric':'Absolute error of complete paired late-frame mean displacement against the original fixed target; no partial means or imputation.','source_hashes':source_hashes,'clips':clips,'limits':['Secondary detector does not retroactively supply online feedback.','Same numeric confidence is not equal calibration across detectors.','YOLO was used earlier in the project; not wholly untouched.','Condition-guided association is not actor identity truth.','No generation or original-result changes.']}
save(OUT/'protocol.json',protocol)
torch.set_num_threads(4);model=YOLO(str(WEIGHT));raw={}
for h,path in sorted(images.items()):
 pred=model.predict(source=path,verbose=False,save=False,**protocol['prediction'])[0]
 raw[h]=[{'label':str(pred.names[int(cl)]),'score':float(sc),'box':[float(v) for v in box]} for box,sc,cl in zip(pred.boxes.xyxy.cpu().tolist(),pred.boxes.conf.cpu().tolist(),pred.boxes.cls.cpu().tolist())]
save(OUT/'raw_detections.json',raw)
associations={key:[{'frame':j,**associate(raw[fr['sha256']],fr['projected_box'])} for j,fr in enumerate(clip['frames'])] for key,clip in clips.items()}
rows=[]
for m,f in methods:
 d=m['direction'];a=associations[d];b=associations['reference'];dx=[center(x['target']['box'])-center(y['target']['box']) if x['target'] and y['target'] else None for x,y in zip(a,b)]
 measurable=all(x is not None for x in dx[4:]);observed=statistics.mean(dx[4:]) if measurable else None;target=f['contract']['target_dx_px']
 rows.append({'direction':d,'method':m['method'],'iteration':0,'controller_status':f['status'],'yolo_measurable':measurable,'frame_deltas':dx,'target_mean_dx_px':target,'observed_mean_dx_px':observed,'absolute_mean_error_px':abs(target-observed) if measurable else None})
for p,h in source_hashes.items():assert sha(p)==h
for h,p in images.items():assert sha(p)==h
assert sha(WEIGHT)==protocol['weight_sha256']
result={'complete':True,'generation_run':False,'unique_png_inferences':len(images),'methods':rows,'association':associations,'elapsed_seconds':time.monotonic()-start,'protocol_sha256':sha(OUT/'protocol.json'),'raw_detection_sha256':sha(OUT/'raw_detections.json'),'limits':protocol['limits']}
save(OUT/'results.json',result)
for key,arr in associations.items():print(key,'associated',[x['association_count'] for x in arr],flush=True)
for r in rows:print(r['direction'],r['method'],r['yolo_measurable'],r['absolute_mean_error_px'],flush=True)
print('SECONDARY_DIAGNOSTIC_COMPLETE',len(images),result['elapsed_seconds'],flush=True)
