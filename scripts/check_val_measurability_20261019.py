"""Source measurability admission for the already frozen validation-split shortlist."""
import json,pickle,time,fcntl,subprocess,hashlib
from pathlib import Path
import torch,numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from dreamer_datasets import load_dataset,boxes3d_utils
from driveloop.native_phase import digest
from driveloop.native_phase_png import FixedBusDetector,associate
ROOT=Path('/home/lesliton/projects/DriveLoop');IN=ROOT/'outputs/revision_20261019_val_source_projection';OUT=ROOT/'outputs/revision_20261019_val_source_measurability';PLAN=ROOT/'outputs/revision_20261019_measurable_source_plan/protocol.json'
read=lambda p:json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
lock=open('/tmp/driveloop-m2-motion-guard.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
OUT.mkdir(exist_ok=False);start=time.monotonic();torch.set_num_threads(4)
plan=read(PLAN);summary=read(IN/'summary.json');assert digest(PLAN)==read(IN/'protocol.json')['amended_plan_sha256']
assert digest(IN/'selected_candidates.pkl')==summary['selected_pickle_sha256']
selected=pickle.loads((IN/'selected_candidates.pkl').read_bytes());assert len(selected)==3
source=Path(plan['source_dataset']);yp=read(ROOT/'outputs/revision_20261019_candidate5614_yolo/protocol.json')
FW=Path('/data/models/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth');YW=ROOT/'yolov8x.pt'
assert digest(FW)==plan['primary_source_measurability']['weight_sha256'];assert digest(YW)==yp['weight_sha256']
fixed={str(p):digest(p) for p in (PLAN,IN/'protocol.json',IN/'summary.json',IN/'selected_candidates.pkl',source/'labels/data.pkl')}
assert fixed[str(source/'labels/data.pkl')]==read(IN/'protocol.json')['source_sha256']
protocol={'written_before_inference':True,'generation_run':False,'script_sha256':digest(__file__),'fixed_hashes':fixed,'source_dataset':str(source),'primary_gate':plan['primary_source_measurability'],'secondary_role':plan['secondary_detector'],'yolo_prediction':yp['prediction'],'yolo_weight_sha256':digest(YW),'fasterrcnn_weight_sha256':digest(FW),'candidate_order':[c['candidate_index'] for c in selected],'geometry_and_runtime_not_yet_verified':True}
save(OUT/'protocol.json',protocol)
data=load_dataset(str(source));resize=transforms.Resize((256,448),interpolation=transforms.InterpolationMode.BILINEAR);frames={};candidateinfo={}
for c in selected:
 n=c['candidate_index'];assert c['source_dataset']==str(source) and c['scene_token'] not in plan['excluded_scenes']
 dest=OUT/f'val_candidate{n}';dest.mkdir();frames[n]=[]
 candidateinfo[n]={'scene_token':c['scene_token'],'instance_token':c['instance_token'],'scene_description':c['scene_description']}
 for j in range(8):
  idx=c['source_camera_starts'][1]+j*3;r=c['groups'][1][j*3];sample=data[idx]
  assert r['cam_type']==sample['cam_type']=='cam_front' and sample['scene_token']==r['scene_token']==c['scene_token']
  assert sample['cam_token']==r['cam_token'];k=r['instance_tokens'].index(c['instance_token']);assert r['ori_labels3d'][k]=='vehicle.bus.rigid'
  im=sample['image'].convert('RGB');assert im.size==(1600,900);im=resize(im);p=dest/f'frame{j}.png';im.save(p)
  cs=boxes3d_utils.boxes3d_to_corners3d(r['boxes3d'][k:k+1],rot_axis=1);assert cs[...,2].min()>.5
  pts=boxes3d_utils.corners3d_to_corners2d(cs,cam_intrinsic=r['calib']['cam_intrinsic'])[0].astype(np.float64)*np.array([448/1600,256/900]);box=np.r_[pts.min(0),pts.max(0)].tolist()
  frames[n].append({'frame':j,'dataset_index':idx,'cam_token':r['cam_token'],'projected_box':box,'png':str(p),'png_sha256':digest(p)})
del data
save(OUT/'frames.json',frames)
raw={str(n):{'fasterrcnn':[],'yolo':[]} for n in frames};model=FixedBusDetector(FW)
for n,fs in frames.items():
 for f in fs:raw[str(n)]['fasterrcnn'].append(model(Image.open(f['png']).convert('RGB')))
del model;torch.cuda.empty_cache();model=YOLO(str(YW))
for n,fs in frames.items():
 for f in fs:
  pred=model.predict(source=f['png'],verbose=False,save=False,**yp['prediction'])[0]
  raw[str(n)]['yolo'].append([{'label':str(pred.names[int(cl)]),'score':float(sc),'box':[float(v) for v in box]} for box,sc,cl in zip(pred.boxes.xyxy.cpu().tolist(),pred.boxes.conf.cpu().tolist(),pred.boxes.cls.cpu().tolist())])
rows=[]
for n,fs in frames.items():
 assoc={name:[associate(ds,f['projected_box']) for ds,f in zip(arr,fs)] for name,arr in raw[str(n)].items()}
 flags={name:[d['target'] is not None for d in arr] for name,arr in assoc.items()}
 rows.append({'candidate_index':n,'source_split':'cam_all_val_v0.0.2',**candidateinfo[n],'primary_pass':all(flags['fasterrcnn'][4:]),'valid_frames':flags,'association':assoc,'geometry_passed':None,'runtime_passed':None})
for p,h in fixed.items():assert digest(p)==h,p
for fs in frames.values():
 for f in fs:assert digest(f['png'])==f['png_sha256']
assert digest(FW)==protocol['fasterrcnn_weight_sha256'] and digest(YW)==protocol['yolo_weight_sha256']
save(OUT/'raw_detections.json',raw)
result={'complete':True,'generation_run':False,'candidate_count':len(rows),'image_count':24,'detector_image_calls':48,'rows':rows,'passed_primary_source_measurability':[r['candidate_index'] for r in rows if r['primary_pass']],'generation_ready':False,'elapsed_seconds':time.monotonic()-start,'protocol_sha256':digest(OUT/'protocol.json'),'limits':['Conditional primary-detector source selection, not unbiased scene coverage.','YOLO is diagnostic only and cannot rescue or reject primary eligibility.','Geometry and runtime admission remain pending; source measurability does not guarantee output measurability.']}
save(OUT/'results.json',result)
for r in rows:print('SOURCE_MEASURABILITY',r['candidate_index'],r['primary_pass'],r['valid_frames'],flush=True)
print('VAL_SOURCE_MEASURABILITY_COMPLETE',result['elapsed_seconds'],flush=True)
