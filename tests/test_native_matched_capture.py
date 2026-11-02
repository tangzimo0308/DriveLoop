import copy,unittest
from scripts.run_native_matched_capture import index_cases,validate_row,ARMS

def rows():return [{'candidate_index':5596,'seed_offset':s,'arm':a} for s in (0,1) for a in ARMS]
def fixture():
 expected={k:{'sha256':'fixed-'+k} for k in ('prompt_embed','img_cond','grounding_downsampler_input','box_downsampler_input')}
 row={'candidate_index':5596,'seed_offset':0,'arm':'original','generation':{'metadata':{'dd2_override_json':{'available':False},'dd2_prompt':None,'dd2_runtime_input_audit':copy.deepcopy(expected)}},'latent_audit':{'initial_noise':{'sha256':'noise0'}},'video_sha256':'exact','exact_video_replay':True,'video_comparison':{'all_decoded_pixels_equal':True}}
 return row,copy.deepcopy(row),expected
class GateTests(unittest.TestCase):
 def test_six_case_inventory(self):self.assertEqual(len(index_cases(rows())),6)
 def test_missing_case_rejected(self):
  with self.assertRaises(AssertionError):index_cases(rows()[:-1])
 def test_duplicate_rejected(self):
  with self.assertRaises(AssertionError):index_cases(rows()+[rows()[0]])
 def test_wrong_candidate_rejected(self):
  x=rows();x[0]['candidate_index']=91
  with self.assertRaises(AssertionError):index_cases(x)
 def test_exact_row_passes(self):validate_row(*fixture())
 def test_changed_video_rejected(self):
  a,b,e=fixture();a['video_sha256']='changed'
  with self.assertRaises(AssertionError):validate_row(a,b,e)
 def test_changed_seed_rejected(self):
  a,b,e=fixture();a['seed_offset']=1
  with self.assertRaises(AssertionError):validate_row(a,b,e)
 def test_wrong_arm_rejected(self):
  a,b,e=fixture();a['arm']='advanced'
  with self.assertRaises(AssertionError):validate_row(a,b,e)
 def test_changed_initial_noise_rejected(self):
  a,b,e=fixture();a['latent_audit']['initial_noise']['sha256']='noise1'
  with self.assertRaises(AssertionError):validate_row(a,b,e)
 def test_changed_input_rejected(self):
  a,b,e=fixture();a['generation']['metadata']['dd2_runtime_input_audit']['box_downsampler_input']['sha256']='wrong'
  with self.assertRaises(AssertionError):validate_row(a,b,e)
 def test_pixel_mismatch_rejected(self):
  a,b,e=fixture();a['video_comparison']['all_decoded_pixels_equal']=False
  with self.assertRaises(AssertionError):validate_row(a,b,e)
if __name__=='__main__':unittest.main()
