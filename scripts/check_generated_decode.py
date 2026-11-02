"""Decode the same captured FINAL generated latent. No encoder, diffusion or MP4."""
import ast,hashlib,inspect,json,time,types
from pathlib import Path
import numpy as np,torch,diffusers
from PIL import Image,ImageDraw
from diffusers import AutoencoderKLTemporalDecoder
from diffusers.image_processor import VaeImageProcessor
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2,FasterRCNN_ResNet50_FPN_V2_Weights
from scripts.dd2_capture_hooks import sha,save
ROOT=Path('/home/lesliton/projects/DriveLoop');OUT=ROOT/'outputs/revision_20261008_generated_decode';OUT.mkdir(exist_ok=False)
RUN=ROOT/'outputs/native_generation_capture_20261008_131957_51857';cap=RUN/'candidate5596_seed0_original/capture'
torch.set_num_threads(4);start=time.monotonic()
def sig(x):
 a=x.detach().cpu().contiguous();return {'shape':list(a.shape),'dtype':str(a.dtype),'sha256':hashlib.sha256(a.numpy().tobytes()).hexdigest(),'finite':bool(torch.isfinite(a).all())}
hook=json.loads((cap/'hook_state.json').read_text());replay=json.loads((RUN/'summary.json').read_text());assert replay['exact_video_replay']
for name,h in hook['files'].items():assert sha(cap/name)==h,name
assert sha(replay['video'])==replay['video_sha256']
z_cpu=torch.load(cap/'final_latents.pt',map_location='cpu',weights_only=True);captured_decoded=torch.load(cap/'decoded_before_postprocess.pt',map_location='cpu',weights_only=True)
assert sig(z_cpu)==hook['final_latents.pt'];assert sig(captured_decoded)==hook['decoded_before_postprocess.pt']
pipe=ROOT/'dreamer-models/dreamer_models/pipelines/drivedreamer2/pipeline_drivedreamer2.py';original=json.loads((ROOT/'outputs/native_multiscene_20261007_214304_13198/protocol.json').read_text())
assert sha(pipe)==original['preflight']['fixed_hashes'][str(pipe)]
base=Path('/data/projects/DriveLoop/hf_cache/stable-video-diffusion-img2vid-xt-1-1');weight=base/'vae/diffusion_pytorch_model.fp16.safetensors';config=base/'vae/config.json'
replay_protocol=json.loads((RUN/'protocol.json').read_text())
for path,h in replay_protocol['vae_hashes'].items():assert sha(path)==h,path
priorpath=ROOT/'outputs/revision_20261008_generated_codec/results.json';prior=json.loads(priorpath.read_text())['arms']['lossless_generated']
plans=json.loads((ROOT/'outputs/revision_20261007_native_multiscene_analysis/results.json').read_text())['plans']
detector=Path('/data/models/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth');old_protocol=json.loads((priorpath.parent/'protocol.json').read_text());assert sha(detector)==old_protocol['detector']['weight_sha256']
assert torch.__version__==hook['runtime']['torch'] and diffusers.__version__==hook['runtime']['diffusers']
assert torch.backends.cudnn.allow_tf32==hook['runtime']['cudnn_allow_tf32'] and torch.backends.cuda.matmul.allow_tf32==hook['runtime']['matmul_allow_tf32']
flags={'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'float32_matmul_precision':torch.get_float32_matmul_precision()}
protocol={'question':'How do decode chunk size and precision affect this actual generated latent under a frozen detector?',
 'written_before_decoding':True,'candidate':5596,'arm':'original','seed_offset':0,'capture_dir':str(cap),'replay_summary_sha256':sha(RUN/'summary.json'),'hook_state_sha256':sha(cap/'hook_state.json'),
 'final_latent_signature':sig(z_cpu),'final_latent_file_sha256':sha(cap/'final_latents.pt'),
 'baseline_gate':'Require EXACT decoded float tensor and all full panorama/front PNGs to match capture, then require all detector outputs to match last analysis; stop on baseline failure.',
 'arms':['fp16_chunk1','fp16_chunk2','fp16_chunk4','fp16_chunk8','strict_fp32_chunk1'],
 'control':'Same final denoised scaled latent, same fp16 VAE checkpoint values. For fp32, exact upcast of latent and weights only; chunk1, TF32 off, float32 matmul precision highest. No re-encoding of images.',
 'decoding':'Actual extracted decode_latents and tensor2vid; clip_latentsFalse; unscale inside decode_latents; num_frames passed to temporal decoder equals chunk length. Full six-view panorama then front cropx448:896.',
 'display':'All arms compared as lossless PNG after same VaeImageProcessor; no MP4, resizing of panorama, noise sampling, diffusion or encoder inference.',
 'detector':old_protocol['detector'],'precision_flags_before':flags,'vae_weight_sha256':sha(weight),'vae_config_sha256':sha(config),'pipeline_sha256':sha(pipe),'script_sha256':sha(__file__),
 'failures':'Record OOM/nonfinite output without fallback or parameter retries; baseline mismatch stops interpretation.',
 'scope':'One previously registered 8-frame generated sequence. Frames not independent scenes; detection coverage does not establish control accuracy, human quality or general improvement. Keep original scores and pipeline unchanged.'}
save(OUT/'protocol.json',protocol)
ns=dict(globals());tree=ast.parse(pipe.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DriveDreamer2Pipeline');decode=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='decode_latents');post=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='tensor2vid');exec(compile(ast.Module(body=[decode,post],type_ignores=[]),str(pipe),'exec'),ns)
processor=VaeImageProcessor(vae_scale_factor=8)
vae=AutoencoderKLTemporalDecoder.from_pretrained(str(base),subfolder='vae',variant='fp16',torch_dtype=torch.float16,local_files_only=True).eval().to('cuda')
weight_values={k:v.detach().cpu().clone() for k,v in vae.state_dict().items()};z=z_cpu.to('cuda');outputs={};audits={};failures={}

def run_arm(arm,latent,chunk):
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t0=time.monotonic()
 try:
  with torch.inference_mode():
   decoded=ns['decode_latents'](types.SimpleNamespace(vae=vae),latent,num_frames=8,decode_chunk_size=chunk,clip_latents=False)
   assert decoded.shape==(1,3,8,256,2688)
   if not bool(torch.isfinite(decoded).all()):raise ValueError('Nonfinite decoded tensor')
   images=ns['tensor2vid'](decoded,processor,output_type='pil')[0]
  torch.cuda.synchronize();assert len(images)==8
  audits[arm]={'latent':sig(latent),'decoded':sig(decoded),'chunk':chunk,'seconds':time.monotonic()-t0,'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'cuda_peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20}
  if arm=='fp16_chunk1':
   eq=torch.equal(decoded.detach().cpu(),captured_decoded);delta=(decoded.detach().cpu()-captured_decoded).abs()
   audits[arm]['captured_decoded_exact']=eq;audits[arm]['decoded_max_abs_difference']=float(delta.max());save(OUT/'decode_audit.json',{'arms':audits,'failures':failures})
   assert eq,'Baseline float output differs from captured original; stop before comparing alternatives'
  (OUT/arm).mkdir();outputs[arm]=[]
  for i,im in enumerate(images):
   im.save(OUT/arm/f'panorama{i}.png');front=im.crop((448,0,896,256));front.save(OUT/arm/f'frame{i}.png');outputs[arm].append(np.asarray(front))
   if arm=='fp16_chunk1':
    assert sha(OUT/arm/f'panorama{i}.png')==hook['files'][f'generated_panorama/frame{i}.png']
    assert sha(OUT/arm/f'frame{i}.png')==hook['files'][f'generated_front/frame{i}.png']
  print('DECODED',arm,'seconds',round(audits[arm]['seconds'],2),'exact_baseline',audits[arm].get('captured_decoded_exact'),flush=True)
 except (torch.OutOfMemoryError,ValueError) as exc:
  failures[arm]={'type':type(exc).__name__,'message':str(exc)};print('ARM_FAILED',arm,failures[arm],flush=True)
 finally:
  torch.cuda.empty_cache();save(OUT/'decode_audit.json',{'arms':audits,'failures':failures})
for chunk in (1,2,4,8):
 run_arm(f'fp16_chunk{chunk}',z,chunk)
 assert 'fp16_chunk1' in outputs,'Baseline must succeed'
torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
vae.float();assert all(torch.equal(v.detach().cpu(),weight_values[k].float()) for k,v in vae.state_dict().items())
z32=z.float();assert torch.equal(z32.half(),z)
run_arm('strict_fp32_chunk1',z32,1)
save(OUT/'precision_audit.json',{'weights_exact_upcast':True,'latent_exact_upcast':True,'strict_flags':{'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'float32_matmul_precision':torch.get_float32_matmul_precision()}})
assert sig(z)==hook['final_latents.pt']
del vae,z,z32,weight_values,captured_decoded;torch.cuda.empty_cache()
torch.set_float32_matmul_precision(flags['float32_matmul_precision']);torch.backends.cuda.matmul.allow_tf32=flags['matmul_allow_tf32'];torch.backends.cudnn.allow_tf32=flags['cudnn_allow_tf32']
model=fasterrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None);model.load_state_dict(torch.load(detector,map_location='cpu',weights_only=True));model.eval().to('cuda');labels=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.meta['categories']
def iou(a,b):
 area=lambda x:max(0,x[2]-x[0])*max(0,x[3]-x[1]);inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));u=area(a)+area(b)-inter;return inter/u if u else 0
results={};sheet=Image.new('RGB',(448*4,288*len(outputs)),'white');draw=ImageDraw.Draw(sheet)
for row,(arm,images) in enumerate(outputs.items()):
 records=[]
 for i,im in enumerate(images):
  x=torch.from_numpy(np.array(im)).permute(2,0,1).to('cuda').float()/255
  with torch.inference_mode():pred=model([x])[0]
  ids=(pred['scores']>=.5).nonzero().flatten();raw=[{'label':labels[int(pred['labels'][j])],'score':float(pred['scores'][j]),'box':pred['boxes'][j].cpu().tolist()} for j in ids]
  matches=[d for d in raw if d['label']=='bus' and iou(d['box'],plans[i]['original'])>=.3];target=matches[0] if len(matches)==1 else None
  if arm=='fp16_chunk1':assert raw==prior[i]['raw'] and len(matches)==prior[i]['association_count']
  ref=prior[i]['target'];dx=(target['box'][0]+target['box'][2]-ref['box'][0]-ref['box'][2])/2 if target and ref else None
  records.append({'frame':i,'raw':raw,'association_count':len(matches),'target':target,'detector_dx_vs_captured':dx,'pixel_mae_vs_captured':float(np.abs(im.astype(np.float64)-outputs['fp16_chunk1'][i].astype(np.float64)).mean()),'png_sha256':sha(OUT/arm/f'frame{i}.png')})
  if i in (0,3,4,7):
   col=(0,3,4,7).index(i);pic=Image.fromarray(im);d=ImageDraw.Draw(pic)
   for m in matches:d.rectangle(m['box'],outline='cyan' if len(matches)==1 else 'orange',width=2)
   sheet.paste(pic,(col*448,row*288+32));draw.text((col*448+4,row*288+8),f'{arm} f{i} | bus matches={len(matches)}',fill='black')
 results[arm]=records
 print('DETECTION',arm,'unique',sum(x['target'] is not None for x in records),'scores',[round(x['target']['score'],3) if x['target'] else None for x in records],flush=True)
summary={a:{'unique_bus_frames':sum(r['target'] is not None for r in rs),'matched_frame_indices':[r['frame'] for r in rs if r['target']],
 'zero_association_indices':[r['frame'] for r in rs if r['association_count']==0],'ambiguous_indices':[r['frame'] for r in rs if r['association_count']>1],
 'mean_pixel_mae_vs_captured':float(np.mean([r['pixel_mae_vs_captured'] for r in rs])),'detector_dx_vs_captured':[r['detector_dx_vs_captured'] for r in rs]} for a,rs in results.items()}
save(OUT/'results.json',{'arms':results,'summary':summary,'failures':failures,'baseline_float_png_and_detections_exact':True,'no_diffusion_or_encoding':True,'elapsed_seconds':time.monotonic()-start});save(OUT/'summary.json',summary);sheet.save(OUT/'generated_decode_comparison.jpg',quality=95)
for path,h in replay_protocol['vae_hashes'].items():assert sha(path)==h
assert sha(pipe)==protocol['pipeline_sha256'] and sha(cap/'final_latents.pt')==protocol['final_latent_file_sha256']
print('GENERATED_DECODE_CHECK_COMPLETE',OUT,flush=True)
