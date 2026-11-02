"""Export existing frames only for a label-masked human measurement check."""
import json,hashlib
from pathlib import Path
import cv2
from dreamer_datasets import load_dataset
ROOT=Path('/home/lesliton/projects/DriveLoop');run=ROOT/'outputs/native_multiscene_20261007_214304_13198';out=ROOT/'outputs/revision_20261008_manual_frames';out.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=json.loads((run/'summary.json').read_text());p=json.loads((run/'protocol.json').read_text());assert s['complete'] and len(s['cases'])==6
rows=[]
for row in s['cases']:
 path=row['generation']['artifacts']['video'];assert sha(path)==row['video_sha256'];cap=cv2.VideoCapture(path);i=0
 while True:
  ok,frame=cap.read()
  if not ok:break
  assert frame.shape==(784,2688,3) and i<8
  image=cv2.resize(frame,(2688,772),interpolation=cv2.INTER_LINEAR)[-256:,448:896]
  name=f"generated_{row['arm']}_seed{row['seed_offset']}_frame{i}.png";assert cv2.imwrite(str(out/name),image)
  rows.append({'filename':name,'kind':'generated','arm':row['arm'],'seed':row['seed_offset'],'frame':i,'image_sha256':sha(out/name),'video_sha256':row['video_sha256']});i+=1
 cap.release();assert i==8
entry=p['selection']['selected'][0];ds=load_dataset(entry['source']['dataset'])
for i in range(8):
 name=f'source_frame{i}.png';ds[24+i*3]['image'].convert('RGB').resize((448,256)).save(out/name)
 rows.append({'filename':name,'kind':'source','frame':i,'image_sha256':sha(out/name)})
(out/'manifest.json').write_text(json.dumps({'run':str(run),'images':rows,'width':448,'height':256,'no_generation':True,'view':'Generated: exact same approximate784-to772 inverse resize and front crop as fixed detector analysis; source: same448x256 PIL resize as auxiliary source check.'},indent=2)+'\n')
print('EXPORTED',len(rows),'existing frames',flush=True)
