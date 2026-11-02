import json,numpy as np
from pathlib import Path
from PIL import Image,ImageDraw
from dreamer_datasets import load_dataset,boxes3d_utils
root=Path('/home/lesliton/projects/DriveLoop/outputs/revision_20261007_native_multiscene_source')
m=json.loads((root/'manifest.json').read_text())
for e in m['selected']:
 ds=load_dataset(e['source']['dataset']);sheet=Image.new('RGB',(448*4,292*2),'white');text=ImageDraw.Draw(sheet)
 for i in range(8):
  r=ds[24+i*3];im=r['image'].convert('RGB').resize((448,252));d=ImageDraw.Draw(im)
  for arm,col in [('original','blue'),('delayed','red'),('advanced','green')]:
   box=e['arms'][arm]['geometry']['frames'][i]['edited_box'];c=boxes3d_utils.boxes3d_to_corners3d(np.array(box)[None],rot_axis=1);p=boxes3d_utils.corners3d_to_corners2d(c,cam_intrinsic=r['calib']['cam_intrinsic'])[0]*[448/1600,252/900]
   for a,b in ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)):d.line([tuple(p[a]),tuple(p[b])],fill=col,width=1)
  x=(i%4)*448;y=(i//4)*292;sheet.paste(im,(x,y+36));text.text((x+5,y+8),f"candidate{e['source']['candidate_index']} frame{i}: blue original/red delay/green advance",fill='black')
 sheet.save(root/f"candidate{e['source']['candidate_index']}_plan.jpg",quality=95)
