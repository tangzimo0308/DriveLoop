"""Source-only codec sensitivity; no diffusion, VAE inference or new scene generation."""
import ast,json,hashlib,inspect,subprocess,os,copy
from pathlib import Path
import cv2,numpy as np,torch,mmengine,imageio,imageio_ffmpeg
from PIL import Image,ImageDraw
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2,FasterRCNN_ResNet50_FPN_V2_Weights
from dreamer_datasets import load_dataset,boxes3d_utils,ImageVisualizer,image_utils
from driveloop.dd2_override import apply_dd2_override_to_sample,tensor_signature,write_override_audit
from scripts.run_native_bus_control import setup_env
ROOT=Path('/home/lesliton/projects/DriveLoop');out=ROOT/'outputs/revision_20261008_codec_check';out.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,v):(out/name).write_text(json.dumps(v,indent=2)+'\n')
setup_env();torch.set_num_threads(4)
run=ROOT/'outputs/native_multiscene_20261007_214304_13198';p=json.loads((run/'protocol.json').read_text());entry=p['selection']['selected'][0]
old=ROOT/'outputs/revision_20261007_native_multiscene_analysis';prior=json.loads((old/'source_detector_check.json').read_text());plans=json.loads((old/'results.json').read_text())['plans']
checkpoint=Path('/data/models/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth')
assert sha(checkpoint)==prior['model_weight_sha256']
transpath=ROOT/'dreamer-train/projects/DriveDreamer2/drivedreamer2/drivedreamer2_transforms.py'
drawpath=ROOT/'dreamer-train/projects/DriveDreamer2/drivedreamer2/drivedreamer2_utils.py'
ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
protocol={'question':'Do export transformations alone destroy source bus detection under the unchanged fixed detector?',
 'before_detector_results':True,'source_candidate':5596,'frames':8,'fps':4,'confidence':.5,'unique_bus_association_iou':.3,'weight_sha256':sha(checkpoint),
 'arms':{'source_direct':'unchanged448x256 PIL-resized real source','scale_only':'actual source-based2688x772 composite -> FFmpeg default scale2688x784 RGB raw -> cv2 linear inverse772 -> front crop; no lossy codec','codec_only':'same composite -> imageio mimsave fps4 macro_block_size1 (no height scale) -> OpenCV decode -> front crop','codec_and_scale':'same composite -> actual imageio mimsave defaults fps4 (macroblock16) -> OpenCV decode -> cv2 inverse772 -> front crop'},
 'composition':'Actual current draw_mv_video function; all6 source images substituted for generated output row; image/map/box panels from actual transform.',
 'limits':'Source-content surrogate, not the unavailable lossless generated frames. Tests RGB scaling and codec pathways on this source, cannot isolate model/VAE/codec interactions on real generated outputs. No changed detection threshold or retrospective replacement of main results.',
 'versions':{'imageio':imageio.__version__,'imageio_ffmpeg':imageio_ffmpeg.__version__,'opencv':cv2.__version__},
 'ffmpeg':ffmpeg,'ffmpeg_sha256':sha(ffmpeg),'ffmpeg_version':subprocess.check_output([ffmpeg,'-version'],text=True).splitlines()[0],
 'writer_signature':str(inspect.signature(imageio_ffmpeg.write_frames)),'script_sha256':sha(__file__),'transform_sha256':sha(transpath),'draw_utils_sha256':sha(drawpath)}
save('protocol.json',protocol)
ns=dict(globals(),HEIGHT=900,WIDTH=1600)
cls=next(n for n in ast.parse(transpath.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='DriveDreamer2_Transform');exec(compile(ast.Module(body=[cls],type_ignores=[]),str(transpath),'exec'),ns)
t=ns['DriveDreamer2_Transform'](resolution=(256,448),gd_input_name='image_hdmap',bd_input_name='image_box',is_train=False,embed_map_path='/data/projects/DriveLoop/clip_text_transform_after_pool_panoramic.pkl',text_embed_dim=1024,box_normal=True)
fn=next(n for n in ast.parse(drawpath.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='draw_mv_video');exec(compile(ast.Module(body=[fn],type_ignores=[]),str(drawpath),'exec'),ns)
ds=load_dataset(entry['source']['dataset']);batch={k:[] for k in ('image','image_box','image_hdmap')}
for cam in range(6):
 for i in range(8):
  r=t(ds[cam*24+i*3])
  batch['image'].append(r['image']);batch['image_box'].append(torch.as_tensor(r['image_box']));batch['image_hdmap'].append(r['image_hdmap'])
panoramas=[image_utils.concat_images([batch['image'][cam*8+i] for cam in range(6)],pad=0) for i in range(8)]
composite=[np.asarray(im.convert('RGB')) for im in ns['draw_mv_video'](panoramas,batch)]
assert all(im.shape==(772,2688,3) for im in composite)
direct=[np.asarray(batch['image'][8+i].convert('RGB')) for i in range(8)]
for i in range(8):
 assert np.array_equal(composite[i][-256:,448:896],direct[i])
 oldpng=np.asarray(Image.open(ROOT/f'outputs/revision_20261008_manual_frames/source_frame{i}.png').convert('RGB'));assert np.array_equal(oldpng,direct[i])
 for im in (direct[i],composite[i]):assert im.dtype==np.uint8
views={'source_direct':direct}
cmd=[ffmpeg,'-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','2688x772','-r','4','-i','-','-vf','scale=2688:784','-f','rawvideo','-pix_fmt','rgb24','-']
q=subprocess.run(cmd,input=b''.join(im.tobytes() for im in composite),stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
assert len(q.stdout)==8*784*2688*3
scaled=np.frombuffer(q.stdout,np.uint8).reshape(8,784,2688,3)
views['scale_only']=[cv2.resize(im,(2688,772),interpolation=cv2.INTER_LINEAR)[-256:,448:896] for im in scaled]
encodings={}
for arm,kwargs in [('codec_only',{'macro_block_size':1}),('codec_and_scale',{})]:
 path=out/(arm+'.mp4');imageio.mimsave(path,composite,fps=4,**kwargs)
 cap=cv2.VideoCapture(str(path));frames=[];h=772 if arm=='codec_only' else 784;assert abs(cap.get(cv2.CAP_PROP_FPS)-4)<1e-6
 while True:
  ok,im=cap.read()
  if not ok:break
  assert im.shape==(h,2688,3);im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
  if h!=772:im=cv2.resize(im,(2688,772),interpolation=cv2.INTER_LINEAR)
  frames.append(im[-256:,448:896])
 cap.release();assert len(frames)==8;views[arm]=frames
 encodings[arm]={'path':str(path),'sha256':sha(path),'decoded_height':h,'fps':4}
save('encoding_audit.json',{'encodings':encodings,'source_front_exactly_matches_prior_pngs':True,'composite_front_exactly_matches_source':True,'scale_only_command':cmd})
print('SOURCE_CODEC_ARMS_READY; no diffusion run',flush=True)
model=fasterrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None);model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval().to('cuda')
labels=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.meta['categories']
def iou(a,b):
 area=lambda x:max(0,x[2]-x[0])*max(0,x[3]-x[1]);inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));u=area(a)+area(b)-inter;return inter/u if u else 0
result={};sheet=Image.new('RGB',(448*2,288*4),'white');draw=ImageDraw.Draw(sheet)
for row,(arm,images) in enumerate(views.items()):
 records=[];(out/arm).mkdir()
 for i,im in enumerate(images):
  Image.fromarray(im).save(out/arm/f'frame{i}.png');x=torch.from_numpy(np.array(im)).permute(2,0,1).to('cuda').float()/255
  with torch.inference_mode():pred=model([x])[0]
  ids=(pred['scores']>=.5).nonzero().flatten();raw=[{'label':labels[int(pred['labels'][j])],'score':float(pred['scores'][j]),'box':pred['boxes'][j].cpu().tolist()} for j in ids]
  candidates=[d for d in raw if d['label']=='bus' and iou(d['box'],plans[i]['original'])>=.3];target=candidates[0] if len(candidates)==1 else None
  diff=im.astype(np.float64)-direct[i].astype(np.float64)
  rec={'frame':i,'raw':raw,'association_count':len(candidates),'target':target,'pixel_mae_vs_direct':float(np.abs(diff).mean()),'pixel_rmse_vs_direct':float(np.sqrt(np.mean(diff**2)))}
  if arm=='source_direct':assert raw==prior['frames'][i]['raw'] and len(candidates)==prior['frames'][i]['association_count']
  ref=prior['frames'][i]['target'];rec['detector_center_shift_vs_direct']=(target['box'][0]+target['box'][2]-ref['box'][0]-ref['box'][2])/2 if target and ref else None
  records.append(rec)
  if i in (0,7):
   pic=Image.fromarray(im);d=ImageDraw.Draw(pic)
   if target:d.rectangle(target['box'],outline='cyan',width=2)
   col=0 if i==0 else 1;sheet.paste(pic,(col*448,row*288+32));draw.text((col*448+4,row*288+8),f'{arm} frame{i} | bus count={len(candidates)}',fill='black')
 result[arm]=records;print(arm,'unique bus frames',sum(r['target'] is not None for r in records),'scores',[round(r['target']['score'],3) if r['target'] else None for r in records],flush=True)
save('results.json',{'arms':result,'direct_detection_exactly_reproduces_prior':True,'no_diffusion_or_vae_inference':True})
sheet.save(out/'source_codec_comparison.jpg',quality=95)
print('CODEC_CHECK_COMPLETE',out,flush=True)
