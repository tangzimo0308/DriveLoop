"""Post-run auxiliary detector check on real source frames, unchanged thresholds."""
import json,hashlib
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw
from dreamer_datasets import load_dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2,FasterRCNN_ResNet50_FPN_V2_Weights
ROOT=Path('/home/lesliton/projects/DriveLoop');out=ROOT/'outputs/revision_20261007_native_multiscene_analysis'
r=json.loads((out/'results.json').read_text());p=json.loads((ROOT/'outputs/native_multiscene_20261007_214304_13198/protocol.json').read_text());ds=load_dataset(p['selection']['selected'][0]['source']['dataset'])
def iou(a,b):
 area=lambda x:max(0,x[2]-x[0])*max(0,x[3]-x[1]);inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));u=area(a)+area(b)-inter;return inter/u if u else 0
checkpoint=Path('/data/models/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth');torch.set_num_threads(4)
model=fasterrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None);model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval().to('cuda');labels=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.meta['categories'];rows=[]
sheet=Image.new('RGB',(448*4,288*2),'white');draw=ImageDraw.Draw(sheet)
for i in range(8):
 im=ds[24+i*3]['image'].convert('RGB').resize((448,256));x=torch.from_numpy(np.array(im)).permute(2,0,1).to('cuda').float()/255
 with torch.inference_mode():pred=model([x])[0]
 ids=(pred['scores']>=.5).nonzero().flatten();raw=[{'label':labels[int(pred['labels'][j])],'score':float(pred['scores'][j]),'box':pred['boxes'][j].cpu().tolist()} for j in ids];candidates=[d for d in raw if d['label']=='bus' and iou(d['box'],r['plans'][i]['original'])>=.3]
 rows.append({'frame':i,'association_count':len(candidates),'target':candidates[0] if len(candidates)==1 else None,'raw':raw});d=ImageDraw.Draw(im);d.rectangle(r['plans'][i]['original'],outline='purple',width=1)
 for det in candidates:d.rectangle(det['box'],outline='blue',width=2)
 x=(i%4)*448;y=(i//4)*288;sheet.paste(im,(x,y+32));draw.text((x+4,y+8),f'real source frame{i}; bus associations={len(candidates)}',fill='black')
sheet.save(out/'source_detector_check.jpg',quality=95)
(out/'source_detector_check.json').write_text(json.dumps({'scope':'Post-run auxiliary source check. Does not replace or change main generated-video analysis. Source resized directly to448x256; generated images additionally pass VAE/video encoding, so not a pixel-matched comparison.','model_weight_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'confidence':.5,'association_iou':.3,'frames':rows},indent=2)+'\n')
print('REAL_SOURCE_ASSOCIATIONS',[(d['frame'],d['association_count']) for d in rows],flush=True)
