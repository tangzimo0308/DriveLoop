"""Observation must preserve tensors, RNG and returned objects; mismatches must fail."""
import json,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np,torch
from PIL import Image
from scripts.dd2_capture_hooks import Capture
from scripts.run_native_generation_capture import compare_videos

class CaptureTests(unittest.TestCase):
 def test_capture_is_observational_and_complete(self):
  with tempfile.TemporaryDirectory() as td:
   c=Capture(Path(td)/'capture');calls=[];z=torch.zeros((1,8,4,32,336),dtype=torch.float16)
   decoded=torch.zeros((1,3,8,256,2688));images=[[Image.new('RGB',(2688,256),(100+i,110,120)) for i in range(8)]]
   class Pipeline:
    def decode_latents(self,latents,num_frames,decode_chunk_size=14,clip_latents=False):
     calls.append(('decode',latents is z,num_frames,decode_chunk_size,clip_latents));return decoded
   owner=Pipeline();owner.vae=types.SimpleNamespace(parameters=lambda:iter([torch.zeros(1,dtype=torch.float16)]),config=types.SimpleNamespace(scaling_factor=.18215))
   def post(v,p,output_type='np'):
    calls.append(('post',v is decoded,output_type));return images
   module=types.SimpleNamespace(DriveDreamer2Pipeline=Pipeline,tensor2vid=post)
   c.patch_pipeline(module);rng=c.rng_sig()
   result=owner.decode_latents(z,8,1,False);self.assertIs(result,decoded)
   result2=module.tensor2vid(result,None,output_type='pil');self.assertIs(result2,images)
   composites=[np.concatenate([np.zeros((516,2688,3),dtype=np.uint8),np.asarray(im)],axis=0) for im in images[0]]
   tester=types.SimpleNamespace(draw_mv_video=lambda im,b:composites);c.patch_tester(tester)
   self.assertIs(tester.draw_mv_video(images[0],{}),composites)
   self.assertEqual(rng,c.rng_sig());self.assertEqual(calls,[('decode',True,8,1,False),('post',True,'pil')])
   self.assertTrue(torch.equal(z,torch.load(c.out/'final_latents.pt',weights_only=True)))
   self.assertTrue(torch.equal(decoded,torch.load(c.out/'decoded_before_postprocess.pt',weights_only=True)))
   state=json.loads((c.out/'hook_state.json').read_text());self.assertEqual(state['counts'],{'decode':1,'postprocess':1,'composite':1})
   self.assertTrue(state['generated_equals_composite_bottom']);self.assertEqual(len(state['files']),26)
   self.assertEqual(state['capture_rng_unchanged'],['before_decode','after_decode','after_postprocess','after_composite'])
   with self.assertRaises(AssertionError):owner.decode_latents(z,8,1,False)
 def test_wrong_decoder_setting_rejected_before_original(self):
  with tempfile.TemporaryDirectory() as td:
   c=Capture(Path(td)/'cap');original=[]
   class P:
    def decode_latents(self,*args):original.append(1)
   m=types.SimpleNamespace(DriveDreamer2Pipeline=P,tensor2vid=lambda *a:None);c.patch_pipeline(m)
   with self.assertRaises(AssertionError):P().decode_latents(torch.zeros(1,8,4,32,336),8,2,False)
   self.assertEqual(original,[])
 def test_rng_mutating_observer_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   c=Capture(Path(td)/'cap')
   with self.assertRaises(AssertionError):c.observe('bad',lambda:torch.rand(1))
   self.assertEqual(c.state['capture_rng_unchanged'],[])
 def test_composite_pixel_change_rejected(self):
  import hashlib
  with tempfile.TemporaryDirectory() as td:
   c=Capture(Path(td)/'cap');c.state['counts']['postprocess']=1
   c.panorama_pixels=[hashlib.sha256(np.zeros((256,2688,3),dtype=np.uint8).tobytes()).hexdigest()]*8
   c.front_pixels=[np.zeros((256,448,3),dtype=np.uint8)]*8
   bad=np.zeros((772,2688,3),dtype=np.uint8);bad[-1,-1]=255
   with self.assertRaises(AssertionError):c.after_composite([bad]*8)
   self.assertEqual(c.state['counts']['composite'],0)

class VideoTests(unittest.TestCase):
 def compare(self,one,two):
  class Reader:
   def __init__(self,frames):self.frames=iter(frames)
   def read(self):
    x=next(self.frames,None);return (x is not None,x)
   def release(self):pass
  with patch('cv2.VideoCapture',side_effect=[Reader(one),Reader(two)]):return compare_videos('old','new')
 def test_exact_pixels_pass(self):
  a=np.zeros((784,2688,3),dtype=np.uint8);self.assertTrue(self.compare([a]*8,[a]*8)['all_decoded_pixels_equal'])
 def test_single_pixel_mismatch_is_not_hidden(self):
  a=np.zeros((784,2688,3),dtype=np.uint8);b=a.copy();b[50,50,0]=1
  r=self.compare([a]*8,[a]*7+[b]);self.assertFalse(r['all_decoded_pixels_equal']);self.assertEqual(r['frames'][-1]['differing_values'],1)
 def test_missing_frame_rejected(self):
  a=np.zeros((784,2688,3),dtype=np.uint8)
  with self.assertRaises(AssertionError):self.compare([a]*8,[a]*7)
 def test_extra_frame_rejected(self):
  a=np.zeros((784,2688,3),dtype=np.uint8)
  with self.assertRaises(AssertionError):self.compare([a]*9,[a]*8)

if __name__=='__main__':
 torch.set_num_threads(2);unittest.main()
