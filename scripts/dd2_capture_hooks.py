"""Process-local observation hooks. Original model files and return values are untouched."""
import functools,hashlib,importlib.abc,importlib.machinery,json,os,pickle,random,sys
from pathlib import Path
PIPE='dreamer_models.pipelines.drivedreamer2.pipeline_drivedreamer2'
TESTER='drivedreamer2.drivedreamer2_tester'

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def save(path,obj):
 Path(path).write_text(json.dumps(obj,indent=2)+'\n')

class Capture:
 def __init__(self,out):
  self.out=Path(out);self.out.mkdir(parents=True,exist_ok=False)
  self.state={'installed':[],'counts':{'decode':0,'postprocess':0,'composite':0},'capture_rng_unchanged':[],'files':{}}
  self.flush()
 def flush(self):save(self.out/'hook_state.json',self.state)
 def tensor_sig(self,x):
  import torch
  a=x.detach().cpu().contiguous()
  return {'shape':list(a.shape),'dtype':str(a.dtype),'sha256':hashlib.sha256(a.numpy().tobytes()).hexdigest(),'finite':bool(torch.isfinite(a).all())}
 def rng_sig(self):
  import numpy as np,torch
  states={'python':pickle.dumps(random.getstate()),'numpy':pickle.dumps(np.random.get_state()),'torch_cpu':torch.get_rng_state().numpy().tobytes()}
  if torch.cuda.is_initialized():
   for i,x in enumerate(torch.cuda.get_rng_state_all()):states[f'torch_cuda_{i}']=x.cpu().numpy().tobytes()
  return {k:hashlib.sha256(v).hexdigest() for k,v in states.items()}
 def observe(self,name,action):
  before=self.rng_sig();result=action();after=self.rng_sig()
  assert before==after,('Capture changed RNG',name)
  self.state['capture_rng_unchanged'].append(name);self.flush();return result
 def file(self,name):self.state['files'][name]=sha(self.out/name)
 def save_tensor(self,name,x):
  import torch
  self.state[name]=self.tensor_sig(x);assert self.state[name]['finite'],name
  torch.save(x.detach().cpu().contiguous(),self.out/name);self.file(name)
 def before_decode(self,owner,z,num_frames,decode_chunk_size,clip_latents):
  assert self.state['counts']['decode']==0,'Expected exactly one decode'
  assert list(z.shape)==[1,8,4,32,336] and num_frames==8 and decode_chunk_size==1 and clip_latents is False
  self.state['decode_settings']={'num_frames':num_frames,'decode_chunk_size':decode_chunk_size,'clip_latents':clip_latents,'vae_dtype':str(next(owner.vae.parameters()).dtype),'scaling_factor':float(owner.vae.config.scaling_factor)}
  self.save_tensor('final_latents.pt',z)
  import torch
  self.state['runtime']={'torch':torch.__version__,'diffusers':getattr(sys.modules.get('diffusers'),'__version__',None),'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32}
 def after_decode(self,z,decoded):
  assert self.tensor_sig(z)==self.state['final_latents.pt'],'Decoder mutated source latent'
  assert list(decoded.shape)==[1,3,8,256,2688]
  self.save_tensor('decoded_before_postprocess.pt',decoded)
  self.state['counts']['decode']+=1
 def after_post(self,video,images,output_type):
  import numpy as np
  assert self.state['counts']['decode']==1 and self.state['counts']['postprocess']==0
  assert output_type=='pil' and len(images)==1 and len(images[0])==8
  assert self.tensor_sig(video)==self.state['decoded_before_postprocess.pt']
  self.front_pixels=[];self.panorama_pixels=[]
  (self.out/'generated_panorama').mkdir();(self.out/'generated_front').mkdir()
  for i,im in enumerate(images[0]):
   assert im.size==(2688,256) and im.mode=='RGB'
   a=np.asarray(im);self.panorama_pixels.append(hashlib.sha256(a.tobytes()).hexdigest());self.front_pixels.append(a[:,448:896].copy())
   im.copy().save(self.out/f'generated_panorama/frame{i}.png')
   im.crop((448,0,896,256)).save(self.out/f'generated_front/frame{i}.png')
   assert hashlib.sha256(np.asarray(im).tobytes()).hexdigest()==self.panorama_pixels[-1]
   self.file(f'generated_panorama/frame{i}.png');self.file(f'generated_front/frame{i}.png')
  self.state['panorama_rgb_sha256']=self.panorama_pixels;self.state['counts']['postprocess']+=1
 def after_composite(self,images):
  import numpy as np
  from PIL import Image
  assert self.state['counts']['postprocess']==1 and self.state['counts']['composite']==0
  assert len(images)==8;(self.out/'composite_before_mp4').mkdir()
  for i,im in enumerate(images):
   a=np.asarray(im);assert a.shape==(772,2688,3) and a.dtype==np.uint8
   assert hashlib.sha256(a[-256:].tobytes()).hexdigest()==self.panorama_pixels[i]
   assert np.array_equal(a[-256:,448:896],self.front_pixels[i])
   Image.fromarray(a.copy()).save(self.out/f'composite_before_mp4/frame{i}.png');self.file(f'composite_before_mp4/frame{i}.png')
  self.state['generated_equals_composite_bottom']=True;self.state['counts']['composite']+=1
 def patch_pipeline(self,module):
  original_decode=module.DriveDreamer2Pipeline.decode_latents;original_post=module.tensor2vid;capture=self
  @functools.wraps(original_decode)
  def decode(owner,latents,num_frames,decode_chunk_size=14,clip_latents=False):
   capture.observe('before_decode',lambda:capture.before_decode(owner,latents,num_frames,decode_chunk_size,clip_latents))
   decoded=original_decode(owner,latents,num_frames,decode_chunk_size,clip_latents)
   capture.observe('after_decode',lambda:capture.after_decode(latents,decoded))
   return decoded
  @functools.wraps(original_post)
  def post(video,processor,output_type='np'):
   images=original_post(video,processor,output_type=output_type)
   capture.observe('after_postprocess',lambda:capture.after_post(video,images,output_type))
   return images
  module.DriveDreamer2Pipeline.decode_latents=decode;module.tensor2vid=post
  self.state['installed'].append(PIPE);self.flush()
 def patch_tester(self,module):
  original=module.draw_mv_video;capture=self
  @functools.wraps(original)
  def draw(images,batch_dict):
   result=original(images,batch_dict)
   capture.observe('after_composite',lambda:capture.after_composite(result))
   return result
  module.draw_mv_video=draw;self.state['installed'].append(TESTER);self.flush()

class ObservingLoader:
 def __init__(self,wrapped,callback):self.wrapped=wrapped;self.callback=callback
 def create_module(self,spec):return self.wrapped.create_module(spec)
 def exec_module(self,module):self.wrapped.exec_module(module);self.callback(module)

class Finder(importlib.abc.MetaPathFinder):
 def __init__(self,capture):self.callbacks={PIPE:capture.patch_pipeline,TESTER:capture.patch_tester}
 def find_spec(self,fullname,path=None,target=None):
  if fullname not in self.callbacks:return None
  spec=importlib.machinery.PathFinder.find_spec(fullname,path)
  assert spec is not None and spec.loader is not None,fullname
  spec.loader=ObservingLoader(spec.loader,self.callbacks[fullname]);return spec

def install(out):
 assert PIPE not in sys.modules and TESTER not in sys.modules,'Hooks must be armed before model imports'
 capture=Capture(out);sys.meta_path.insert(0,Finder(capture));return capture
