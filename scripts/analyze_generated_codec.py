"""Fixed-detector analysis of actual generated frames before/after saving. No diffusion."""
import json,hashlib,time,subprocess
from pathlib import Path
import cv2,numpy as np,torch,imageio,imageio_ffmpeg
from PIL import Image,ImageDraw
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2,FasterRCNN_ResNet50_FPN_V2_Weights
from scripts.dd2_capture_hooks import sha,save
ROOT=Path('/home/lesliton/projects/DriveLoop');RUN=ROOT/'outputs/native_generation_capture_20261008_131957_51857';OUT=ROOT/'outputs/revision_20261008_generated_codec';OUT.mkdir(exist_ok=False)
torch.set_num_threads(4);start=time.monotonic()
s=json.loads((RUN/'summary.json').read_text());assert s['exact_video_replay'] and s['all_decoded_pixels_equal'] and s['video_sha256_match']
capdir=Path(s['capture_dir']);hooks=json.loads((capdir/'hook_state.json').read_text())
for name,h in hooks['files'].items():assert sha(capdir/name)==h,name
assert sha(s['video'])==s['video_sha256']==s['reference_video_sha256']
priorpath=ROOT/'outputs/revision_20261007_native_multiscene_analysis/results.json';old=json.loads(priorpath.read_text());plans=old['plans'];prior=next(r for r in old['cases'] if r['seed']==0 and r['arm']=='original');assert prior['video_sha256']==s['video_sha256']
sourceaudit=json.loads((ROOT/'outputs/revision_20261007_native_multiscene_analysis/source_detector_check.json').read_text());checkpoint=Path('/data/models/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth');assert sha(checkpoint)==sourceaudit['model_weight_sha256']
ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
protocol={'question':'Does the actual video saving pipeline change bus detection in candidate5596 original seed0?',
 'written_before_detection':True,'source_run':str(RUN),'source_summary_sha256':sha(RUN/'summary.json'),'exact_video_replay_verified':True,'video_sha256':s['video_sha256'],
 'arms':{'lossless_generated':'Captured actual tensor2vid front PNG before video saving, 448x256','scale_only':'Captured complete composite772 high -> FFmpeg RGB scale784 -> cv2 inverse772 -> front crop, no lossy codec','codec_only':'Captured complete composite -> same imageio codec with macro_block_size1 (772 high) -> cv2 decode -> front crop','actual_mp4':'Unchanged captured MP4784 high -> cv2 inverse772 -> front crop; exact prior analysis path'},
 'validation':'Verify all captured file hashes and bottom-row/panorama/front pixel equality. Re-encode full default path and require MP4 SHA equality. All prior original-seed0 detector outputs must match exactly.',
 'detector':{'model':'FasterRCNN_ResNet50_FPN_V2 COCO_V1','confidence':.5,'unique_bus_iou':.3,'weight_sha256':sha(checkpoint)},
 'prior_results_sha256':sha(priorpath),'ffmpeg_sha256':sha(ffmpeg),'versions':{'torch':torch.__version__,'cv2':cv2.__version__,'imageio':imageio.__version__,'imageio_ffmpeg':imageio_ffmpeg.__version__},
 'script_sha256':sha(__file__),'claim_limit':'One registered generated clip, 8 correlated frames. Detection association is not physical category/actor existence/trajectory success. RGB-scale-only does not exactly isolate all YUV operations within codec; no additive causal decomposition or retrospective score substitution.'}
save(OUT/'protocol.json',protocol)
composite=[];direct=[]
for i in range(8):
 im=np.array(Image.open(capdir/f'composite_before_mp4/frame{i}.png').convert('RGB'));front=np.array(Image.open(capdir/f'generated_front/frame{i}.png').convert('RGB'));pan=np.array(Image.open(capdir/f'generated_panorama/frame{i}.png').convert('RGB'))
 assert im.shape==(772,2688,3) and pan.shape==(256,2688,3) and front.shape==(256,448,3)
 assert np.array_equal(im[-256:],pan) and np.array_equal(pan[:,448:896],front)
 composite.append(im);direct.append(front)
views={'lossless_generated':direct}
cmd=[ffmpeg,'-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','2688x772','-r','4','-i','-','-vf','scale=2688:784','-f','rawvideo','-pix_fmt','rgb24','-']
proc=subprocess.run(cmd,input=b''.join(x.tobytes() for x in composite),stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
assert len(proc.stdout)==8*784*2688*3
scaled=np.frombuffer(proc.stdout,np.uint8).reshape(8,784,2688,3)
views['scale_only']=[cv2.resize(x,(2688,772),interpolation=cv2.INTER_LINEAR)[-256:,448:896] for x in scaled]
imageio.mimsave(OUT/'codec_only.mp4',composite,fps=4,macro_block_size=1)
imageio.mimsave(OUT/'default_reencoded.mp4',composite,fps=4)
assert sha(OUT/'default_reencoded.mp4')==s['video_sha256'],'Re-encoded default MP4 does not match captured video'
for arm,path,h in [('codec_only',OUT/'codec_only.mp4',772),('actual_mp4',Path(s['video']),784)]:
 cap=cv2.VideoCapture(str(path));assert abs(cap.get(cv2.CAP_PROP_FPS)-4)<1e-6;frames=[]
 for i in range(8):
  ok,im=cap.read();assert ok and im.shape==(h,2688,3)
  # Match earlier analysis: inverse resize in BGR, then RGB conversion and front crop.
  if h==784:im=cv2.resize(im,(2688,772),interpolation=cv2.INTER_LINEAR)
  frames.append(cv2.cvtColor(im[-256:,448:896],cv2.COLOR_BGR2RGB))
 assert cap.read()[0] is False;cap.release();views[arm]=frames
save(OUT/'encoding_audit.json',{'capture_hashes_all_verified':True,'composite_generated_pixels_equal':True,'default_reencoding_sha256':sha(OUT/'default_reencoded.mp4'),'default_reencoding_exactly_matches_original':True,'codec_only_sha256':sha(OUT/'codec_only.mp4'),'scale_only_command':cmd})
model=fasterrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None);model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval().to('cuda');labels=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.meta['categories']
def iou(a,b):
 area=lambda x:max(0,x[2]-x[0])*max(0,x[3]-x[1]);inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));u=area(a)+area(b)-inter;return inter/u if u else 0
results={};sheet=Image.new('RGB',(448*4,288*4),'white');draw=ImageDraw.Draw(sheet)
for row,(arm,images) in enumerate(views.items()):
 (OUT/arm).mkdir();records=[]
 for i,im in enumerate(images):
  Image.fromarray(im).save(OUT/arm/f'frame{i}.png');x=torch.from_numpy(np.array(im)).permute(2,0,1).to('cuda').float()/255
  with torch.inference_mode():pred=model([x])[0]
  ids=(pred['scores']>=.5).nonzero().flatten();raw=[{'label':labels[int(pred['labels'][j])],'score':float(pred['scores'][j]),'box':pred['boxes'][j].cpu().tolist()} for j in ids]
  matches=[d for d in raw if d['label']=='bus' and iou(d['box'],plans[i]['original'])>=.3];target=matches[0] if len(matches)==1 else None
  if arm=='actual_mp4':assert raw==prior['frames'][i]['raw'] and len(matches)==prior['frames'][i]['association_count']
  delta=im.astype(np.float64)-direct[i].astype(np.float64)
  rec={'frame':i,'raw':raw,'association_count':len(matches),'target':target,'pixel_mae_vs_lossless':float(np.abs(delta).mean()),'png_sha256':sha(OUT/arm/f'frame{i}.png')}
  ref=results['lossless_generated'][i]['target'] if 'lossless_generated' in results else target
  rec['detector_dx_vs_lossless']=(target['box'][0]+target['box'][2]-ref['box'][0]-ref['box'][2])/2 if target and ref else None
  records.append(rec)
  if i in (0,3,4,7):
   col=(0,3,4,7).index(i);pic=Image.fromarray(im);d=ImageDraw.Draw(pic)
   if target:d.rectangle(target['box'],outline='cyan',width=2)
   sheet.paste(pic,(448*col,288*row+32));draw.text((448*col+4,288*row+8),f'{arm} f{i} | bus matches={len(matches)}',fill='black')
 results[arm]=records
 print('DETECTION',arm,'unique',sum(x['target'] is not None for x in records),'scores',[round(x['target']['score'],3) if x['target'] else None for x in records],flush=True)
summary={a:{'unique_bus_frames':sum(r['target'] is not None for r in rs),'matched_frame_indices':[r['frame'] for r in rs if r['target']],
 'zero_association_frames':sum(r['association_count']==0 for r in rs),'ambiguous_frames':sum(r['association_count']>1 for r in rs),'mean_pixel_mae_vs_lossless':float(np.mean([r['pixel_mae_vs_lossless'] for r in rs])),
 'detector_dx_vs_lossless':[r['detector_dx_vs_lossless'] for r in rs]} for a,rs in results.items()}
transitions=[]
for i in range(8):
 a=results['lossless_generated'][i];b=results['actual_mp4'][i]
 transitions.append({'frame':i,'lossless_unique':a['target'] is not None,'mp4_unique':b['target'] is not None,'dx_if_both':b['detector_dx_vs_lossless']})
save(OUT/'results.json',{'arms':results,'summary':summary,'lossless_to_mp4_transitions':transitions,'all_prior_mp4_detections_exactly_reproduced':True,'no_diffusion_or_vae_inference':True,'elapsed_seconds':time.monotonic()-start})
save(OUT/'summary.json',summary);sheet.save(OUT/'generated_codec_comparison.jpg',quality=95)
print('GENERATED_CODEC_ANALYSIS_COMPLETE',OUT,flush=True)
