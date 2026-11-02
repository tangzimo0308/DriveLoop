"""Prespecified full six-case analysis: lossless primary, MP4 sensitivity."""
import json,time,hashlib,pickle
from pathlib import Path
import numpy as np,torch,cv2
from PIL import Image,ImageDraw
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2,FasterRCNN_ResNet50_FPN_V2_Weights
from dreamer_datasets import boxes3d_utils
from scripts.dd2_capture_hooks import sha,save
from scripts.run_candidate1583_capture import index_cases,check_files,ARMS
from scripts.run_native_matched_capture import validate_hooks
from scripts.run_native_advance_pair import check_pair
ROOT=Path('/home/lesliton/projects/DriveLoop');RUN=ROOT/'outputs/candidate1583_capture_20261009_084734_12216';OUT=ROOT/'outputs/revision_20261009_candidate1583_analysis';OUT.mkdir(exist_ok=False)
start=time.monotonic();torch.set_num_threads(4)
s=json.loads((RUN/'summary.json').read_text());protocol=json.loads((RUN/'protocol.json').read_text());assert s['complete'] and s['all_captures_validated'] and s['paired_input_checks_passed'] and s['new_render_count']==6
rows=index_cases(s['cases']);pre=protocol['preflight'];check_files(pre);case=pre['cases'][0];assert case['candidate_index']==1583 and len(pre['cases'])==1
assert sha(ROOT/'outputs/revision_20261008_candidate1583_inputs/preflight.json')==protocol['preflight_sha256']
for (seed,arm),row in rows.items():
 assert row['scene_token']==case['scene_token'];meta=row['generation']['metadata']
 assert meta['dd2_override_json']['available'] is False and meta['dd2_prompt'] is None
 for k in ('prompt_embed','img_cond','grounding_downsampler_input','box_downsampler_input'):assert meta['dd2_runtime_input_audit'][k]['sha256']==case['runtime_audits'][arm][k]['sha256']
 validate_hooks(Path(row['capture_dir']));assert sha(row['generation']['artifacts']['video'])==row['video_sha256']
 assert row['latent_audit']['scale']==1 and row['latent_audit']['image_latent_before']==row['latent_audit']['image_latent_after']
 if arm!='original':check_pair(rows[(seed,'original')],row)
assert rows[(0,'original')]['latent_audit']['initial_noise']['sha256']!=rows[(1,'original')]['latent_audit']['initial_noise']['sha256']
plans=[{} for _ in range(8)]
for arm in ARMS:
 records=pickle.loads((Path(case['inputs'][arm]['dataset'])/'labels/data.pkl').read_bytes());token=case['inputs'][arm]['instance_token']
 for i in range(8):
  r=records[24+i*3];j=r['instance_tokens'].index(token);cs=boxes3d_utils.boxes3d_to_corners3d(r['boxes3d'][j:j+1],rot_axis=1);assert cs[...,2].min()>.5
  pts=boxes3d_utils.corners3d_to_corners2d(cs,cam_intrinsic=r['calib']['cam_intrinsic'])[0].astype(np.float64)*np.array([448/1600,256/900]);plans[i][arm]=np.r_[pts.min(0),pts.max(0)].tolist()
for arm in ('delayed','advanced'):
 expected=protocol['selection']['selected'][0]['arms'][arm]['geometry']['frames']
 for i in range(8):
  a=plans[i][arm];b=plans[i]['original'];dx=(a[0]+a[2]-b[0]-b[2])/2
  assert abs(dx-expected[i]['projected_center_shift_x_px_at448'])<1e-3,(arm,i,dx)
detector=Path('/data/models/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth');assert sha(detector)=='dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf'
assert protocol['analysis_plan']['confidence']==.5 and protocol['analysis_plan']['unique_bus_iou']==.3
save(OUT/'protocol.json',{'question':'Under the prespecified fixed detector, how much paired displacement is measurable in candidate1583?', 'written_before_detection':True,'run':str(RUN),'run_summary_sha256':sha(RUN/'summary.json'),'capture_protocol_sha256':sha(RUN/'protocol.json'),'prespecified_analysis_plan':protocol['analysis_plan'],'script_sha256':sha(__file__),'detector_weight_sha256':sha(detector),'frame_count_per_format':48,'primary':'png','secondary':'mp4','late_indices':[4,5,6,7],'all_input_and_capture_hashes_verified':True,'planned_projections_match_geometry':True,'limits':'One selected native-bus scene, two seeds, dependent same-seed paired contrasts; box-center movement is not physical ground truth or closed-loop efficacy.'})
images={}
for (seed,arm),row in rows.items():
 png=[np.array(Image.open(Path(row['capture_dir'])/f'generated_front/frame{i}.png').convert('RGB')) for i in range(8)]
 assert all(x.shape==(256,448,3) for x in png);images[('png',seed,arm)]=png
 cap=cv2.VideoCapture(row['generation']['artifacts']['video']);mp4=[]
 for i in range(8):
  ok,im=cap.read();assert ok and im.shape==(784,2688,3)
  restored=cv2.resize(im,(2688,772),interpolation=cv2.INTER_LINEAR);mp4.append(cv2.cvtColor(restored[-256:,448:896],cv2.COLOR_BGR2RGB))
 assert cap.read()[0] is False;cap.release();images[('mp4',seed,arm)]=mp4
model=fasterrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None);model.load_state_dict(torch.load(detector,map_location='cpu',weights_only=True));model.eval().to('cuda');labels=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.meta['categories']
def iou(a,b):
 area=lambda x:max(0,x[2]-x[0])*max(0,x[3]-x[1]);inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));u=area(a)+area(b)-inter;return inter/u if u else 0
cx=lambda d:(d['box'][0]+d['box'][2])/2
center=lambda b:(b[0]+b[2])/2
records={};cases=[]
for seed in (0,1):
 for arm in ARMS:
  for fmt in ('png','mp4'):
   rs=[];dest=OUT/f'{fmt}_seed{seed}_{arm}';dest.mkdir()
   for i,im in enumerate(images[(fmt,seed,arm)]):
    Image.fromarray(im).save(dest/f'frame{i}.png')
    x=torch.from_numpy(im.copy()).permute(2,0,1).to('cuda').float()/255
    with torch.inference_mode():pred=model([x])[0]
    ids=(pred['scores']>=.5).nonzero().flatten();raw=[{'label':labels[int(pred['labels'][j])],'score':float(pred['scores'][j]),'box':pred['boxes'][j].cpu().tolist()} for j in ids]
    candidates=[d for d in raw if d['label']=='bus' and iou(d['box'],plans[i][arm])>=.3]
    row={'frame':i,'raw':raw,'association_count':len(candidates),'target':candidates[0] if len(candidates)==1 else None,'png_sha256':sha(dest/f'frame{i}.png')}
    rs.append(row)
   records[(fmt,seed,arm)]=rs;cases.append({'format':fmt,'seed':seed,'arm':arm,'frames':rs})
   print('DETECTED',fmt,seed,arm,'unique',sum(d['target'] is not None for d in rs),'counts',[d['association_count'] for d in rs],flush=True)
del model;torch.cuda.empty_cache()
pairs=[]
for fmt in ('png','mp4'):
 for seed in (0,1):
  for arm in ('delayed','advanced'):
   fs=[]
   for i in range(8):
    a=records[(fmt,seed,'original')][i]['target'];b=records[(fmt,seed,arm)][i]['target'];dx=cx(b)-cx(a) if a and b else None
    fs.append({'frame':i,'expected_dx':center(plans[i][arm])-center(plans[i]['original']),'observed_dx':dx,
     'original_association_count':records[(fmt,seed,'original')][i]['association_count'],'edited_association_count':records[(fmt,seed,arm)][i]['association_count']})
   late=[f['observed_dx'] for f in fs[4:8]];complete=all(x is not None for x in late)
   pair={'format':fmt,'seed':seed,'arm':arm,'frames':fs,'measurable_pair_frames':sum(f['observed_dx'] is not None for f in fs),
    'late_mean':float(np.mean(late)) if complete else None,'late_complete':complete,'late_missing_indices':[f['frame'] for f in fs[4:8] if f['observed_dx'] is None],'expected_late_mean':float(np.mean([f['expected_dx'] for f in fs[4:8]]))}
   pairs.append(pair)
transitions=[]
for seed in (0,1):
 for arm in ARMS:
  for i in range(8):
   a=records[('png',seed,arm)][i];b=records[('mp4',seed,arm)][i]
   transitions.append({'seed':seed,'arm':arm,'frame':i,'png_count':a['association_count'],'mp4_count':b['association_count'],'png_unique':a['target'] is not None,'mp4_unique':b['target'] is not None,
    'mp4_minus_png_center':cx(b['target'])-cx(a['target']) if a['target'] and b['target'] else None,
    'pixel_mae':float(np.abs(images[('png',seed,arm)][i].astype(np.float64)-images[('mp4',seed,arm)][i].astype(np.float64)).mean())})
paired_format_changes=[]
for seed in (0,1):
 for arm in ('delayed','advanced'):
  a=next(p for p in pairs if (p['format'],p['seed'],p['arm'])==('png',seed,arm));b=next(p for p in pairs if (p['format'],p['seed'],p['arm'])==('mp4',seed,arm))
  for x,y in zip(a['frames'],b['frames']):
   dx=x['observed_dx'];dy=y['observed_dx']
   paired_format_changes.append({'seed':seed,'arm':arm,'frame':x['frame'],'png_dx':dx,'mp4_dx':dy,'mp4_minus_png_dx':dy-dx if dx is not None and dy is not None else None,'sign_flip':bool(dx*dy<0) if dx is not None and dy is not None else None})
summary={fmt:{'unique_frames':sum(r['target'] is not None for (f,s,a),rs in records.items() if f==fmt for r in rs),
 'zero_frames':sum(r['association_count']==0 for (f,s,a),rs in records.items() if f==fmt for r in rs),
 'ambiguous_frames':sum(r['association_count']>1 for (f,s,a),rs in records.items() if f==fmt for r in rs),
 'measurable_pair_frames':sum(p['measurable_pair_frames'] for p in pairs if p['format']==fmt),'complete_late_pairs':sum(p['late_complete'] for p in pairs if p['format']==fmt)} for fmt in ('png','mp4')}
summary['png_to_mp4']={'lost_unique':sum(t['png_unique'] and not t['mp4_unique'] for t in transitions),'gained_unique':sum(not t['png_unique'] and t['mp4_unique'] for t in transitions),
 'both_unique':sum(t['png_unique'] and t['mp4_unique'] for t in transitions),'neither_unique':sum(not t['png_unique'] and not t['mp4_unique'] for t in transitions),
 'common_paired_comparisons':sum(t['mp4_minus_png_dx'] is not None for t in paired_format_changes),'paired_sign_flips':sum(t['sign_flip'] is True for t in paired_format_changes)}
result={'cases':cases,'pairs':pairs,'transitions':transitions,'paired_format_changes':paired_format_changes,'summary':summary,'all_input_and_capture_hashes_verified':True,'plans':plans,'no_generation':True,'elapsed_seconds':time.monotonic()-start}
save(OUT/'results.json',result);save(OUT/'summary.json',summary)
for seed in (0,1):
 sheet=Image.new('RGB',(448*4,288*6),'white');draw=ImageDraw.Draw(sheet)
 for a,arm in enumerate(ARMS):
  for f,fmt in enumerate(('png','mp4')):
   row=2*a+f
   for col,i in enumerate((3,4,5,7)):
    im=Image.fromarray(images[(fmt,seed,arm)][i]);d=ImageDraw.Draw(im);rec=records[(fmt,seed,arm)][i]
    matches=[q for q in rec['raw'] if q['label']=='bus' and iou(q['box'],plans[i][arm])>=.3]
    for q in matches:d.rectangle(q['box'],outline='cyan' if len(matches)==1 else 'orange',width=2)
    sheet.paste(im,(col*448,row*288+32));draw.text((col*448+4,row*288+8),f'{fmt} seed{seed} {arm} f{i} matches={len(matches)}',fill='black')
 sheet.save(OUT/f'comparison_seed{seed}.jpg',quality=95)
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
for seed in (0,1):
 for ai,arm in enumerate(('delayed','advanced')):
  ax=axes[seed,ai]
  subset=[q for q in pairs if q['seed']==seed and q['arm']==arm]
  ax.plot(range(8),[f['expected_dx'] for f in subset[0]['frames']],color='black',linestyle='--',label='Planned projected shift')
  for fmt,color,marker in [('png','#166534','o'),('mp4','#9f1239','s')]:
   pair=next(q for q in subset if q['format']==fmt)
   ax.plot(range(8),[np.nan if f['observed_dx'] is None else f['observed_dx'] for f in pair['frames']],color=color,marker=marker,label=f'{fmt.upper()} detector shift')
  ax.axhline(0,color='gray',linewidth=.7);ax.set_title(f'{arm}, seed {seed}');ax.set_xticks(range(8));ax.grid(alpha=.2);ax.set_ylabel('Edited minus original center x (px)');ax.set_xlabel('Frame index');ax.legend(fontsize=8)
fig.suptitle('One fixed scene: missing/ambiguous pairs remain gaps; detector shifts are not physical ground truth',fontsize=10);fig.tight_layout();fig.savefig(OUT/'paired_displacement.png',dpi=180);plt.close(fig)
print('SUMMARY',json.dumps(summary),flush=True)
for pair in pairs:print('LATE_PAIR',pair['format'],pair['seed'],pair['arm'],'mean',pair['late_mean'],'missing',pair['late_missing_indices'],flush=True)
print('CANDIDATE1583_ANALYSIS_COMPLETE',OUT,flush=True)
