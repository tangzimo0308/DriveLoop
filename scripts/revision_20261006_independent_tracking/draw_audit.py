import cv2,json
from pathlib import Path
from PIL import Image,ImageDraw
root=Path('/home/lesliton/projects/DriveLoop')
held=json.loads((root/'outputs/revision_20261006_independent_tracking/results.json').read_text())
yolo=json.loads((root/'outputs/revision_20261006_height_result_audit/detections.json').read_text())['1']['records']
case=next(c for c in held['cases'] if c['mode']=='local_vehicle_bottom_v1' and c['seed']==1)
cap=cv2.VideoCapture(str(root/'outputs/bus_height_pair_20261006_010720_56265/local_vehicle_bottom_v1_seed1/artifacts/iteration_01.mp4'))
frames=[]
while True:
 ok,im=cap.read()
 if not ok:break
 frames.append(cv2.cvtColor(im[-256:,448:896],cv2.COLOR_BGR2RGB))
cap.release()
sheet=Image.new('RGB',(448*4,286*2),'white');d=ImageDraw.Draw(sheet)
for row,detector in enumerate(('YOLO / confidence 0.2','Faster R-CNN / confidence 0.5')):
 for col,f in enumerate((3,5,6,7)):
  im=Image.fromarray(frames[f]);draw=ImageDraw.Draw(im)
  dets=yolo[f]['retained_bus'] if row==0 else case['residual'][f]
  for det in dets:
   draw.rectangle(det['box'],outline='red',width=2)
   score=det.get('score',det.get('confidence'))
   draw.text((int(det['box'][0]),max(0,int(det['box'][1])-12)),f'bus {score:.2f}',fill='red')
  x=col*448;y=row*286;sheet.paste(im,(x,y+26));d.text((x+4,y+4),f'{detector} | seed 1 | frame {f}',fill='black')
sheet.save(root/'outputs/revision_20261006_independent_tracking/detector_comparison.jpg',quality=95)
