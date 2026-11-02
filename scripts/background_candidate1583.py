import json,hashlib
from pathlib import Path
import cv2,numpy as np
ROOT=Path("/home/lesliton/projects/DriveLoop/outputs/revision_20261009_candidate1583_analysis")
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
r=json.loads((ROOT/'results.json').read_text());cs={(c['format'],c['seed'],c['arm']):c for c in r['cases']};rows=[]
for fmt in ('png','mp4'):
 for seed in (0,1):
  for arm in ('delayed','advanced'):
   for i in range(8):
    boxes=[r['plans'][i]['original'],r['plans'][i][arm]]
    for a in ('original',arm):
     t=cs[(fmt,seed,a)]['frames'][i]['target']
     if t:boxes.append(t['box'])
    images=[cv2.imread(str(ROOT/f'{fmt}_seed{seed}_{a}/frame{i}.png')) for a in ('original',arm)]
    b=background_shift(*images,boxes);rows.append({'format':fmt,'seed':seed,'arm':arm,'frame':i,'background':b})
s={}
for fmt in ('png','mp4'):
 b=[x['background'] for x in rows if x['format']==fmt];good=[x for x in b if x['available']]
 s[fmt]={'available':len(good),'total':len(b),'minimum_points':min(x['count'] for x in good) if good else None,'maximum_absolute_median_dx':max(abs(x['median_dx']) for x in good) if good else None}
result={'rows':rows,'summary':s,'definition':'Same-frame original-to-edited forward/backward LK; exclude union of both planned and available detected boxes plus12px. Minimum20 valid features, forward/backward tolerance1px; same settings as earlier source diagnostics.','scope':'Auxiliary background sensitivity only. Median outside masks is not camera-motion ground truth; no adjustment of primary detector results.','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'main_results_sha256':hashlib.sha256((ROOT/'results.json').read_bytes()).hexdigest()}
(ROOT/'background_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(s))
