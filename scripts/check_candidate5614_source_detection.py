"""Matched source-image detectability diagnostic; no generation or retuning."""
import json,pickle,hashlib,time,fcntl,subprocess
from pathlib import Path
import numpy as np,torch,torchvision,ultralytics
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from dreamer_datasets import load_dataset,boxes3d_utils
from driveloop.native_phase import NativePhaseRegistry,digest
from driveloop.native_phase_png import FixedBusDetector,associate,DETECTOR_SHA
ROOT=Path('/home/lesliton/projects/DriveLoop');OUT=ROOT/'outputs/revision_20261019_candidate5614_source'
RUN=ROOT/'outputs/candidate5614_pilot_20261019_011254_9302'
PRE=ROOT/'outputs/revision_20261015_candidate5614_inputs/preflight.json'
PRESHA='a82687280eaf6639c00f9b44ad79f519077a8d6e8c9740e9a372eb3b9528a550'
YP=ROOT/'outputs/revision_20261019_candidate5614_yolo'
FW=Path('/data/models/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth');YW=ROOT/'yolov8x.pt'
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
lock=open('/tmp/driveloop-m2-motion-guard.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
OUT.mkdir(exist_ok=False);(OUT/'source_front').mkdir();start=time.monotonic();torch.set_num_threads(4)
r=NativePhaseRegistry(PRE,PRESHA);assert r.candidate==5614
m=r.case['inputs']['original'];native=Path(m['dataset']);records=pickle.loads((native/'labels/data.pkl').read_bytes());data=load_dataset(str(native))
f_result=RUN/'delayed/direct_target/result.json';prov=read(f_result)['attempt_history'][0]['evaluation']['metrics']['native_phase_feedback']['provenance']
yp=read(YP/'protocol.json');yr=read(YP/'results.json')
assert yr['complete'] and digest(YP/'protocol.json')==yr['protocol_sha256']
assert digest(YW)==yp['weight_sha256'] and digest(FW)==DETECTOR_SHA
assert prov['reference']['registry_sha256']==r.fingerprint
files=[PRE,f_result,YP/'protocol.json',YP/'results.json',native/'labels/data.pkl',native/'images/data.mdb',ROOT/'dreamer-train/projects/DriveDreamer2/drivedreamer2/drivedreamer2_transforms.py',ROOT/'driveloop/native_phase_png.py']
fixed={str(p):digest(p) for p in files};frames=[]
resize=transforms.Resize((256,448),interpolation=transforms.InterpolationMode.BILINEAR)
for j in range(8):
 idx=24+j*3;record=records[idx];sample=data[idx];assert record['cam_type']=='cam_front' and record['scene_token']==r.scene
 k=record['instance_tokens'].index(r.instance);assert record['ori_labels3d'][k]=='vehicle.bus.rigid'
 im=sample['image'].convert('RGB');assert im.size==(1600,900)
 small=resize(im);path=OUT/f'source_front/frame{j}.png';small.save(path)
 corners=boxes3d_utils.boxes3d_to_corners3d(record['boxes3d'][k:k+1],rot_axis=1)
 assert corners[...,2].min()>.5
 pts=boxes3d_utils.corners3d_to_corners2d(corners,cam_intrinsic=record['calib']['cam_intrinsic'])[0].astype(np.float64)*np.array([448/1600,256/900]);box=np.r_[pts.min(0),pts.max(0)].tolist()
 assert np.allclose(box,prov['reference']['frames'][j]['projected_box'],atol=1e-9,rtol=0)
 frames.append({'frame':j,'dataset_index':idx,'cam_type':record['cam_type'],'cam_token':record['cam_token'],'sample_token':record['sample_token'],'frame_idx':int(record['frame_idx']),'instance_token':r.instance,'source_rgb_sha256':hashlib.sha256(np.asarray(im).tobytes()).hexdigest(),'png':str(path),'png_sha256':digest(path),'projected_box':box})
del data
protocol={'written_before_inference':True,'generation_run':False,'script_sha256':digest(__file__),'registry_sha256':r.fingerprint,'scene_token':r.scene,'candidate_index':r.candidate,'instance_token':r.instance,'frames':frames,'source_preprocessing':'Real RGB source frames resized from 1600x900 to 448x256 by torchvision PIL bilinear Resize, matching DD2 input-image resize. No brightness/sharpening enhancement. Generated images unchanged.','fasterrcnn_weight_sha256':digest(FW),'yolo_weight_sha256':digest(YW),'yolo_prediction':yp['prediction'],'association':yp['association'],'fixed_hashes':fixed,'versions':{'torch':torch.__version__,'torchvision':torchvision.__version__,'ultralytics':ultralytics.__version__},'comparison':'Report per-frame source and previously saved generated-reference detections. No per-frame center-error comparison against moving source truth and no target reassignment.','limits':['Source frames are not expected to equal generated frames pixel-wise.','Detection failure is not proof that the annotated actor is absent.','Original generated reference is seed0 only; do not replace it with other seeds.','Retrospective diagnostic, not a new eligibility filter or change to original results.']}
save(OUT/'protocol.json',protocol)
raw={};assoc={}
model=FixedBusDetector(FW)
raw['fasterrcnn']=[model(Image.open(f['png']).convert('RGB')) for f in frames]
del model;torch.cuda.empty_cache()
model=YOLO(str(YW));raw['yolo']=[]
for f in frames:
 pred=model.predict(source=f['png'],verbose=False,save=False,**yp['prediction'])[0]
 raw['yolo'].append([{'label':str(pred.names[int(cl)]),'score':float(sc),'box':[float(v) for v in box]} for box,sc,cl in zip(pred.boxes.xyxy.cpu().tolist(),pred.boxes.conf.cpu().tolist(),pred.boxes.cls.cpu().tolist())])
for name,arr in raw.items():assoc[name]=[{'frame':j,**associate(det,frames[j]['projected_box'])} for j,det in enumerate(arr)]
comparison={}
for name,gen in [('fasterrcnn',prov['reference_detections']),('yolo',yr['association']['reference'])]:
 source=[x['target'] is not None for x in assoc[name]];generated=[x['target'] is not None for x in gen]
 comparison[name]={'source_valid':source,'generated_seed0_valid':generated,'source_total':sum(source),'generated_total':sum(generated),'source_late':sum(source[4:]),'generated_late':sum(generated[4:]),'source_late_complete':all(source[4:]),'generated_late_complete':all(generated[4:]),'source_only_valid_frames':[j for j in range(8) if source[j] and not generated[j]],'generated_only_valid_frames':[j for j in range(8) if generated[j] and not source[j]]}
for p,h in fixed.items():assert digest(p)==h,p
for f in frames:assert digest(f['png'])==f['png_sha256']
r.verify();assert digest(FW)==protocol['fasterrcnn_weight_sha256'] and digest(YW)==protocol['yolo_weight_sha256']
save(OUT/'raw_detections.json',raw)
result={'complete':True,'generation_run':False,'source_frames':8,'detector_image_calls':16,'comparison':comparison,'association':assoc,'elapsed_seconds':time.monotonic()-start,'protocol_sha256':digest(OUT/'protocol.json'),'raw_detection_sha256':digest(OUT/'raw_detections.json'),'limits':protocol['limits']}
save(OUT/'results.json',result)
print(json.dumps(comparison,indent=2));print('SOURCE_DIAGNOSTIC_COMPLETE',result['elapsed_seconds'],flush=True)
