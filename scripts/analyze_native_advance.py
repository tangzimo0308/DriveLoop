"""Analyze existing paired videos only. Detector positions and background-flow sensitivity."""
import json,hashlib
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image,ImageDraw
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2,FasterRCNN_ResNet50_FPN_V2_Weights
from dreamer_datasets import load_dataset,boxes3d_utils
ROOT=Path('/home/lesliton/projects/DriveLoop');run=ROOT/'outputs/native_advance_pair_20261006_214053_102070';out=ROOT/'outputs/revision_20261006_native_advance_analysis';out.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):(out/n).write_text(json.dumps(v,indent=2)+'\n')
s=json.loads((run/'summary.json').read_text());pr=json.loads((run/'protocol.json').read_text());assert s['complete'] and s['paired_input_checks_passed']
m=pr['manifest'];ds=load_dataset(m['native_dataset']);plans=[]
for i,g in enumerate(m['geometry']['frames']):
 r=ds[24+i*3];p={}
 for name,key in [('original_control_reused','original_box'),('advanced_track','edited_box')]:
  c=boxes3d_utils.boxes3d_to_corners3d(np.array(g[key])[None],rot_axis=1)
  pts=boxes3d_utils.corners3d_to_corners2d(c,cam_intrinsic=r['calib']['cam_intrinsic'])[0]*[448/1600,256/900]
  p[name]=[float(pts[:,0].min()),float(pts[:,1].min()),float(pts[:,0].max()),float(pts[:,1].max())]
 plans.append(p)
def iou(a,b):
 area=lambda x:max(0,x[2]-x[0])*max(0,x[3]-x[1]);inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));u=area(a)+area(b)-inter;return inter/u if u else 0
cx=lambda b:(b[0]+b[2])/2
checkpoint=Path('/data/models/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth')
save('protocol.json',{'analysis_of_completed_run':str(run),'model':'FasterRCNN_ResNet50_FPN_V2 COCO_V1','confidence':.5,'weight_sha256':sha(checkpoint),'script_sha256':sha(__file__),
 'view':'same as native-control report: inverse resize784to772, bottom256 x448:896; approximate, not calibrated',
 'association':'bus predictions >=.5, IoU>=.3 with this arms projected box; exactly one required or unmeasured; no highest-score fallback',
 'displacement':'matched detector bounding-box center x, advanced minus original at the same frame and seed; all frames reported',
 'expected_displacement':'projected 3D bounding-rectangle center x difference, not mean corner x',
 'background_sensitivity':'median bidirectionally-consistent LK displacement outside union of expected and detected target rectangles, 12px margin; at least20 points; auxiliary only',
 'claim':'Exploratory diagnostic with analysis method declared before this advance run; not a benchmark preregistration, physical trajectory truth or cut-in success.'})
torch.set_num_threads(4);model=fasterrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None);model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval().to('cuda')
labels=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.meta['categories'];videos={};detected={};allrows=[]
for row in s['cases']:
 seed=row['seed_offset'];arm=row['arm'];path=row['generation']['artifacts']['video'];assert sha(path)==row['video_sha256'];cap=cv2.VideoCapture(path);frames=[];dets=[];i=0
 while True:
  ok,frame=cap.read()
  if not ok:break
  assert frame.shape==(784,2688,3) and i<8
  restored=cv2.resize(frame,(2688,772),interpolation=cv2.INTER_LINEAR);front=restored[-256:,448:896];frames.append(front)
  x=torch.from_numpy(cv2.cvtColor(front,cv2.COLOR_BGR2RGB)).permute(2,0,1).to('cuda').float()/255
  with torch.inference_mode():pred=model([x])[0]
  ids=(pred['scores']>=.5).nonzero().flatten();raw=[{'label':labels[int(pred['labels'][j])],'score':float(pred['scores'][j]),'box':pred['boxes'][j].cpu().tolist()} for j in ids]
  candidates=[d for d in raw if d['label']=='bus' and iou(d['box'],plans[i][arm])>=.3]
  dets.append({'frame':i,'raw':raw,'association_count':len(candidates),'target':candidates[0] if len(candidates)==1 else None});i+=1
 cap.release();assert i==8;videos[(seed,arm)]=frames;detected[(seed,arm)]=dets
 allrows.append({'seed':seed,'arm':arm,'frames':dets,'video_sha256':row['video_sha256']});print('DETECTED',seed,arm,'unique bus frames',sum(d['target'] is not None for d in dets),flush=True)
del model;torch.cuda.empty_cache()

def background_shift(a,b,boxes):
 ga=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY);gb=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY);mask=np.ones(ga.shape,np.uint8)*255
 for box in boxes:
  x0,y0,x1,y1=box;cv2.rectangle(mask,(max(0,int(x0)-12),max(0,int(y0)-12)),(min(447,int(x1)+12),min(255,int(y1)+12)),0,-1)
 p=cv2.goodFeaturesToTrack(ga,maxCorners=500,qualityLevel=.01,minDistance=5,mask=mask)
 if p is None:return {'available':False,'reason':'no_features'}
 q,st,_=cv2.calcOpticalFlowPyrLK(ga,gb,p,None,winSize=(21,21),maxLevel=3)
 if q is None:return {'available':False,'reason':'no_forward_flow'}
 back,sb,_=cv2.calcOpticalFlowPyrLK(gb,ga,q,None,winSize=(21,21),maxLevel=3)
 if back is None:return {'available':False,'reason':'no_backward_flow'}
 good=(st[:,0]==1)&(sb[:,0]==1)&(np.linalg.norm(back[:,0]-p[:,0],axis=1)<=1)
 rounded=np.rint(q[:,0]).astype(int);inside=(rounded[:,0]>=0)&(rounded[:,0]<448)&(rounded[:,1]>=0)&(rounded[:,1]<256);good&=inside
 for j in np.flatnonzero(good):
  if mask[rounded[j,1],rounded[j,0]]==0:good[j]=False
 delta=q[:,0][good]-p[:,0][good]
 if len(delta)<20:return {'available':False,'reason':'too_few_consistent_features','count':len(delta)}
 median=np.median(delta,axis=0);mad=np.median(np.abs(delta-median),axis=0)
 return {'available':True,'count':len(delta),'median_dx':float(median[0]),'median_dy':float(median[1]),'mad_dx':float(mad[0]),'mad_dy':float(mad[1])}

pairs=[];sheet=Image.new('RGB',(448*2,288*4),'white');draw=ImageDraw.Draw(sheet)
for seed in (0,1):
 frames=[]
 for i in range(8):
  a=detected[(seed,'original_control_reused')][i]['target'];b=detected[(seed,'advanced_track')][i]['target'];expected=cx(plans[i]['advanced_track'])-cx(plans[i]['original_control_reused'])
  boxes=list(plans[i].values())+([a['box']] if a else [])+([b['box']] if b else [])
  bg=background_shift(videos[(seed,'original_control_reused')][i],videos[(seed,'advanced_track')][i],boxes)
  observed=cx(b['box'])-cx(a['box']) if a and b else None
  frames.append({'frame':i,'expected_dx':expected,'observed_dx':observed,'background':bg,'background_adjusted_dx':observed-bg['median_dx'] if observed is not None and bg['available'] else None})
  if i in (4,7):
   col=0 if i==4 else 1
   for k,arm in enumerate(('original_control_reused','advanced_track')):
    im=Image.fromarray(cv2.cvtColor(videos[(seed,arm)][i],cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(im);target=detected[(seed,arm)][i]['target']
    d.rectangle(plans[i][arm],outline='purple',width=1)
    if target:d.rectangle(target['box'],outline='blue' if k==0 else 'red',width=2)
    y=(seed*2+k)*288;sheet.paste(im,(col*448,y+32));draw.text((col*448+4,y+8),f'seed{seed} {arm} frame{i}',fill='black')
 pairs.append({'seed':seed,'frames':frames});print('PAIR',seed,[(d['frame'],round(d['expected_dx'],2),None if d['observed_dx'] is None else round(d['observed_dx'],2),round(d['background'].get('median_dx',0),2)) for d in frames],flush=True)
sheet.save(out/'matched_bus_positions.jpg',quality=95)
save('results.json',{'cases':allrows,'pairs':pairs,'plans':plans,'all_frame_associations_unique':all(d['target'] is not None for case in allrows for d in case['frames']),'no_video_generation':True})
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(8,4.5));ax.plot(range(8),[r['expected_dx'] for r in pairs[0]['frames']],color='black',marker='o',label='Planned box-center shift')
for pair in pairs:ax.plot(range(8),[r['observed_dx'] for r in pair['frames']],marker='o',label=f"Detected bus shift, seed {pair['seed']}")
ax.axhline(0,color='gray',linewidth=.8);ax.set(xlabel='Frame index',ylabel='Advanced - original center x (pixels)',title='Paired bus-position response (448-pixel front view)');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(out/'paired_displacement.png',dpi=180);plt.close(fig)
print('ADVANCE_ANALYSIS_COMPLETE',out,flush=True)
