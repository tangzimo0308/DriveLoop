"""Frozen-source VAE round-trip diagnostic. No diffusion, optimizer or MP4."""
import ast,json,hashlib,inspect,os,time,types
from pathlib import Path
import cv2,numpy as np,torch,mmengine,diffusers
from PIL import Image,ImageDraw
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2,FasterRCNN_ResNet50_FPN_V2_Weights
from diffusers import AutoencoderKLTemporalDecoder
from diffusers.image_processor import VaeImageProcessor
from dreamer_datasets import load_dataset,boxes3d_utils
from driveloop.dd2_override import apply_dd2_override_to_sample,tensor_signature,write_override_audit
from scripts.run_native_bus_control import setup_env
ROOT=Path('/home/lesliton/projects/DriveLoop');out=ROOT/'outputs/revision_20261008_vae_check';out.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):(out/n).write_text(json.dumps(v,indent=2)+'\n')
def sig(t):return {'shape':list(t.shape),'dtype':str(t.dtype),'sha256':hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),'finite':bool(torch.isfinite(t).all())}
setup_env();torch.set_num_threads(4);start=time.monotonic()
run=ROOT/'outputs/native_multiscene_20261007_214304_13198';pr=json.loads((run/'protocol.json').read_text());entry=pr['selection']['selected'][0];assert entry['source']['candidate_index']==5596
transpath=ROOT/'dreamer-train/projects/DriveDreamer2/drivedreamer2/drivedreamer2_transforms.py';pipepath=ROOT/'dreamer-models/dreamer_models/pipelines/drivedreamer2/pipeline_drivedreamer2.py';loader=pipepath.parent/'drivedreamer2_loaders.py'
for path in (transpath,pipepath,loader):assert sha(path)==pr['preflight']['fixed_hashes'][str(path)]
base=Path('/data/projects/DriveLoop/hf_cache/stable-video-diffusion-img2vid-xt-1-1')
weight=base/'vae/diffusion_pytorch_model.fp16.safetensors';config=base/'vae/config.json'
checkpoint=Path('/data/models/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth')
old=ROOT/'outputs/revision_20261007_native_multiscene_analysis';prior=json.loads((old/'source_detector_check.json').read_text());plans=json.loads((old/'results.json').read_text())['plans']
assert sha(checkpoint)==prior['model_weight_sha256']
protocol={'question':'Does the actual DD2 VAE round trip preserve source bus detection, before MP4?',
 'written_before_reconstruction':True,'source_candidate':5596,'frames':8,
 'vae_path':str(base/'vae'),'vae_weight_sha256':sha(weight),'vae_config':json.loads(config.read_text()),'vae_config_sha256':sha(config),
 'weight_origin':'Tester loads fp16 variant from local SVD base; inspected DD2 load_weights updates UNet/downsamplers only, not VAE.',
 'dtype':'float16, matching current tester and pipeline; config force_upcast is true but current custom pipeline does not call an upcast path here',
 'input':'actual transform.Resize BILINEAR -> ToTensor -> Normalize0.5, six views concatenated horizontally to256x2688, then half() as current condition encoder',
 'arms':['source_direct','actual_input_pixels','vae_mode','vae_sample1234','vae_sample5678'],'posterior_sampling_seeds':[1234,5678],
 'sampling_scope':'Diagnostic RNG seeds on VAE posterior only, not replay of full generation RNG. Encode every source frame; actual img_cond mode normally encodes only first frame.',
 'decoding':'Extracted actual decode_latents method; scale latent and unscale in decode; decode_chunk_size1, clip_latentsFalse; actual tensor2vid and VaeImageProcessor to PIL.',
 'front_crop':'x448:896 of full six-view panorama, no MP4 or inverse-height-resize',
 'detector':{'model':'FasterRCNN_ResNet50_FPN_V2 COCO_V1','weight_sha256':sha(checkpoint),'confidence':.5,'unique_bus_iou':.3},
 'limits':'Source-posterior round trip, not denoised generated latent. Even successful reconstruction cannot isolate diffusion as sole failure cause or rule out VAE/generation interactions.',
 'script_sha256':sha(__file__),'code_hashes':{str(x):sha(x) for x in (transpath,pipepath,loader)},'torch':torch.__version__,'diffusers':diffusers.__version__}
save('protocol.json',protocol)
ns=dict(globals(),HEIGHT=900,WIDTH=1600);cls=next(n for n in ast.parse(transpath.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='DriveDreamer2_Transform');exec(compile(ast.Module(body=[cls],type_ignores=[]),str(transpath),'exec'),ns)
transform=ns['DriveDreamer2_Transform'](resolution=(256,448),gd_input_name='image_hdmap',bd_input_name='image_box',is_train=False,embed_map_path='/data/projects/DriveLoop/clip_text_transform_after_pool_panoramic.pkl',text_embed_dim=1024,box_normal=True)
tree=ast.parse(pipepath.read_text());pc=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DriveDreamer2Pipeline')
decode=next(n for n in pc.body if isinstance(n,ast.FunctionDef) and n.name=='decode_latents');post=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='tensor2vid')
exec(compile(ast.Module(body=[decode,post],type_ignores=[]),str(pipepath),'exec'),ns)
ds=load_dataset(entry['source']['dataset']);inputs=[];direct=[]
for i in range(8):
 views=[transform.transform(ds[cam*24+i*3]['image'].convert('RGB')) for cam in range(6)]
 x=torch.cat(views,dim=2)[None];assert x.shape==(1,3,256,2688)
 if i==0:
  first_input_float32=sig(x);assert first_input_float32['sha256']==pr['preflight']['cases'][0]['runtime_audits']['original']['img_cond']['sha256']
 inputs.append(x.half())
 direct.append(np.asarray(ds[24+i*3]['image'].convert('RGB').resize((448,256))))
 assert np.array_equal(direct[-1],np.asarray(Image.open(ROOT/f'outputs/revision_20261008_manual_frames/source_frame{i}.png').convert('RGB')))
processor=VaeImageProcessor(vae_scale_factor=8)
input_pixels=[np.asarray(processor.postprocess(x.float(),output_type='pil')[0])[:,448:896] for x in inputs]
save('input_audit.json',{'first_frame_float32_exactly_matches_prior':True,'first_frame_float32':first_input_float32,'first_frame_fp16':sig(inputs[0]),'all_input_signatures':[sig(x) for x in inputs],
 'previous_runtime_input_audit':pr['preflight']['cases'][0]['runtime_audits']['original']['img_cond'],
 'note':'These signatures use fp16 encoder input; prior runtime audit may record pre-cast dtype. PNG input arm includes standard display rounding.'})
vae=AutoencoderKLTemporalDecoder.from_pretrained(str(base),subfolder='vae',variant='fp16',torch_dtype=torch.float16,local_files_only=True).eval().to('cuda')
assert all(p.dtype==torch.float16 for p in vae.parameters())
assert float(vae.config.scaling_factor)==protocol['vae_config']['scaling_factor']
gens={seed:torch.Generator(device='cuda').manual_seed(seed) for seed in (1234,5678)}
latents={arm:[] for arm in ('vae_mode','vae_sample1234','vae_sample5678')};posterior_audit=[]
with torch.inference_mode():
 for i,x in enumerate(inputs):
  posterior=vae.encode(x.to('cuda')).latent_dist
  assert torch.isfinite(posterior.mean).all() and torch.isfinite(posterior.std).all()
  posterior_audit.append({'frame':i,'mean':sig(posterior.mean),'std_mean':float(posterior.std.float().mean())})
  latents['vae_mode'].append(posterior.mode()*vae.config.scaling_factor)
  for seed in gens:latents[f'vae_sample{seed}'].append(posterior.sample(generator=gens[seed])*vae.config.scaling_factor)
 outputs={'source_direct':direct,'actual_input_pixels':input_pixels};latent_audit={}
 for arm,tensors in latents.items():
  z=torch.cat(tensors,dim=0)[None];assert sig(z)['finite'];latent_audit[arm]=sig(z)
  decoded=ns['decode_latents'](types.SimpleNamespace(vae=vae),z,num_frames=8,decode_chunk_size=1,clip_latents=False)
  assert decoded.shape==(1,3,8,256,2688) and torch.isfinite(decoded).all()
  images=ns['tensor2vid'](decoded,processor,output_type='pil')[0];assert len(images)==8
  (out/arm).mkdir()
  outputs[arm]=[]
  for i,im in enumerate(images):
   assert im.size==(2688,256);im.save(out/arm/f'panorama{i}.png');front=im.crop((448,0,896,256));front.save(out/arm/f'frame{i}.png');outputs[arm].append(np.asarray(front))
  print('VAE_RECONSTRUCTED',arm,'elapsed',round(time.monotonic()-start,2),flush=True)
save('latent_audit.json',{'posterior':posterior_audit,'latents':latent_audit,'decoder_chunk_size':1,'no_diffusion':True,'no_mp4':True})
del vae,latents,posterior,decoded,tensors,z;torch.cuda.empty_cache()
model=fasterrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None);model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval().to('cuda');labels=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.meta['categories']
def iou(a,b):
 area=lambda x:max(0,x[2]-x[0])*max(0,x[3]-x[1]);inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));u=area(a)+area(b)-inter;return inter/u if u else 0
results={};sheet=Image.new('RGB',(448*2,288*5),'white');draw=ImageDraw.Draw(sheet)
for row,(arm,images) in enumerate(outputs.items()):
 (out/arm).mkdir(exist_ok=True);records=[]
 for i,im in enumerate(images):
  Image.fromarray(im).save(out/arm/f'frame{i}.png');x=torch.from_numpy(np.array(im)).permute(2,0,1).to('cuda').float()/255
  with torch.inference_mode():pred=model([x])[0]
  ids=(pred['scores']>=.5).nonzero().flatten();raw=[{'label':labels[int(pred['labels'][j])],'score':float(pred['scores'][j]),'box':pred['boxes'][j].cpu().tolist()} for j in ids]
  candidates=[d for d in raw if d['label']=='bus' and iou(d['box'],plans[i]['original'])>=.3];target=candidates[0] if len(candidates)==1 else None
  if arm=='source_direct':assert raw==prior['frames'][i]['raw']
  diff=im.astype(np.float64)-input_pixels[i].astype(np.float64)
  rec={'frame':i,'raw':raw,'association_count':len(candidates),'target':target,'mae_vs_actual_input_pixels':float(np.abs(diff).mean()),'png_sha256':sha(out/arm/f'frame{i}.png')}
  if 'actual_input_pixels' in results:
   ref=results['actual_input_pixels'][i]['target'];rec['detector_dx_vs_input_pixels']=(target['box'][0]+target['box'][2]-ref['box'][0]-ref['box'][2])/2 if target and ref else None
  records.append(rec)
  if i in (0,7):
   pic=Image.fromarray(im);d=ImageDraw.Draw(pic)
   if target:d.rectangle(target['box'],outline='cyan',width=2)
   col=0 if i==0 else 1;sheet.paste(pic,(col*448,row*288+32));draw.text((col*448+4,row*288+8),f'{arm} f{i} | bus associations={len(candidates)}',fill='black')
 results[arm]=records
 print('DETECTION',arm,'unique',sum(r['target'] is not None for r in records),'scores',[round(r['target']['score'],3) if r['target'] else None for r in records],flush=True)
save('results.json',{'arms':results,'source_direct_reproduces_prior':True,'no_diffusion':True,'no_mp4':True,'elapsed_seconds':time.monotonic()-start})
sheet.save(out/'vae_comparison.jpg',quality=95)
assert sha(weight)==protocol['vae_weight_sha256']
for path,h in protocol['code_hashes'].items():assert sha(path)==h
print('VAE_CHECK_COMPLETE',out,flush=True)
