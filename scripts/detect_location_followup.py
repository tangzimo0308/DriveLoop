"""Fixed 16-frame detector replay. Never imports annotations or generates videos."""
import json, hashlib, time, fcntl, subprocess, pickle
from pathlib import Path
import numpy as np
import torch, torchvision, ultralytics
from PIL import Image
from ultralytics import YOLO
from dreamer_datasets import boxes3d_utils
from driveloop.native_phase_png import FixedBusDetector, DETECTOR_SHA

ROOT=Path('/home/lesliton/projects/DriveLoop')
OUT=ROOT/'outputs/revision_20261019_location_followup_detection'
EXPORT=ROOT/'outputs/revision_20261019_location_followup_frames'
FWEIGHT=Path('/data/models/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth')
YWEIGHT=ROOT/'yolov8x.pt'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,o):Path(p).write_text(json.dumps(o,indent=2)+'\n')

lock=open('/tmp/driveloop-m2-motion-guard.lock','a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip(), 'GPU busy; no automatic retry'
start=time.monotonic()
manifest=read(EXPORT/'manifest.json')
sampling=ROOT/'outputs/revision_20261019_location_followup_sampling.json'
assert sha(sampling)==manifest['protocol_sha256']
plan=read(sampling)
assert len(manifest['frames'])==16 and len({f['sha256'] for f in manifest['frames']})==16
assert {(f['scene_key'],f['seed'],f['frame']) for f in manifest['frames']}=={(s,z,j) for s in ('candidate5614','val568') for z in (1,2) for j in (4,5,6,7)}
assert sha(FWEIGHT)==DETECTOR_SHA
prior=read(ROOT/'outputs/revision_20261019_val568_yolo/protocol.json')
assert sha(YWEIGHT)==prior['weight_sha256']=='3df4ada6b4dad6d657868f2fdf7faecfb34dcfccf3a25c4b82079064718524c8'
frozen={str(EXPORT/'manifest.json'):sha(EXPORT/'manifest.json'),str(sampling):sha(sampling)}
frames=[]
for f in manifest['frames']:
    assert f['sha256'] not in plan['known_local_png_hashes']
    assert sha(f['source_path'])==sha(EXPORT/f['filename'])==f['sha256']
    assert sha(f['receipt_path'])==f['receipt_sha256']
    c=read(f['receipt_path']);g=c['generation'];meta=g['metadata'];q=meta['native_phase_contract']
    assert g['iteration']==f['seed'] and q['arm']=='original' and q['phase_s']==0
    assert all(q[k]==f[k] for k in ('scene_token','instance_token','registry_sha256'))
    assert sha(g['artifacts']['video'])==c['video_sha256']
    # Original references have the same authenticated runtime inputs across seeds.
    seed0=Path(f['receipt_path']).parents[1]/'iteration_00_original/capture_receipt.json'
    c0=read(seed0);m0=c0['generation']['metadata']
    assert meta['dd2_runtime_input_audit']==m0['dd2_runtime_input_audit']
    binding=meta['dd2_source_sample_binding'];assert binding['ready']
    labels=Path(binding['labels_path']);records=pickle.loads(labels.read_bytes())
    r=records[24+f['frame']*3];j=r['instance_tokens'].index(q['instance_token'])
    corners=boxes3d_utils.boxes3d_to_corners3d(r['boxes3d'][j:j+1],rot_axis=1)
    assert corners[...,2].min()>.5
    points=boxes3d_utils.corners3d_to_corners2d(corners,cam_intrinsic=r['calib']['cam_intrinsic'])[0].astype(np.float64)*np.array([448/1600,256/900])
    box=np.r_[points.min(0),points.max(0)].tolist()
    oldpath=ROOT/('outputs/revision_20261019_'+f['scene_key']+'_yolo/protocol.json')
    old=read(oldpath)['clips']['reference'];assert old['registry_sha256']==q['registry_sha256']
    assert box==old['frames'][f['frame']]['projected_box']
    for p in (f['receipt_path'],seed0,labels,oldpath):frozen[str(p)]=sha(p)
    with Image.open(f['source_path']) as im:assert im.mode=='RGB' and im.size==(448,256)
    frames.append({**f,'condition_box':box})
OUT.mkdir(exist_ok=False)
protocol={'written_before_inference':True,'script_sha256':sha(__file__),'manifest_sha256':sha(EXPORT/'manifest.json'),'sampling_sha256':sha(sampling),'frozen_rule_sha256':plan['frozen_location_rule_sha256'],
 'FRCNN':{'weight_path':str(FWEIGHT),'weight_sha256':sha(FWEIGHT),'confidence':.5,'device':'cuda','implementation_sha256':sha(ROOT/'driveloop/native_phase_png.py')},
 'YOLO':{'weight_path':str(YWEIGHT),'weight_sha256':sha(YWEIGHT),'prediction':prior['prediction']},
 'versions':{'torch':torch.__version__,'torchvision':torchvision.__version__,'ultralytics':ultralytics.__version__},'input_hashes':frozen,'frames':frames,
 'human_labels_loaded':False,'selection':'All fixed16 frames; no exclusions or replacements','generation_run':False,'limits':plan['limits']}
save(OUT/'protocol.json',protocol)
torch.set_num_threads(4)
raw={f['sha256']:{} for f in frames}
detector=FixedBusDetector(str(FWEIGHT))
for f in frames:
    with Image.open(f['source_path']) as im:raw[f['sha256']]['FRCNN']=detector(im)
print('FRCNN_FINISHED_16',flush=True)
del detector;torch.cuda.empty_cache()
detector=YOLO(str(YWEIGHT))
for f in frames:
    p=detector.predict(source=f['source_path'],verbose=False,save=False,**prior['prediction'])[0]
    raw[f['sha256']]['YOLO']=[{'label':str(p.names[int(cl)]),'score':float(sc),'box':[float(v) for v in box]} for box,sc,cl in zip(p.boxes.xyxy.cpu().tolist(),p.boxes.conf.cpu().tolist(),p.boxes.cls.cpu().tolist())]
print('YOLO_FINISHED_16',flush=True)
for p,h in frozen.items():assert sha(p)==h
for f in frames:assert sha(f['source_path'])==f['sha256']
assert sha(FWEIGHT)==protocol['FRCNN']['weight_sha256'] and sha(YWEIGHT)==protocol['YOLO']['weight_sha256']
save(OUT/'raw_detections.json',raw)
save(OUT/'receipt.json',{'complete':True,'unique_images':16,'detector_image_calls':32,'generation_run':False,'human_labels_loaded':False,'elapsed_seconds':time.monotonic()-start,'protocol_sha256':sha(OUT/'protocol.json'),'raw_detections_sha256':sha(OUT/'raw_detections.json')})
print('FOLLOWUP_DETECTION_COMPLETE',time.monotonic()-start,flush=True)
