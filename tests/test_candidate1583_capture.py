import copy
import pytest
from scripts.run_candidate1583_capture import index_cases,check_files

def cases():return [{'candidate_index':1583,'scene_token':'scene','seed_offset':s,'arm':a} for s in (0,1) for a in ('original','delayed','advanced')]
def test_complete_case_inventory():assert len(index_cases(cases()))==6
@pytest.mark.parametrize('error',['missing','duplicate','candidate','scene','seed','arm'])
def test_inventory_errors_rejected(error):
 r=cases()
 if error=='missing':r.pop()
 elif error=='duplicate':r[-1]=copy.deepcopy(r[0])
 elif error=='candidate':r[0]['candidate_index']=5596
 elif error=='scene':r[0]['scene_token']='other'
 elif error=='seed':r[0]['seed_offset']=2
 elif error=='arm':r[0]['arm']='unknown'
 with pytest.raises(AssertionError):index_cases(r)
def test_frozen_input_tamper_rejected(tmp_path):
 import hashlib
 p=tmp_path/'frozen';p.write_bytes(b'original');h=hashlib.sha256(p.read_bytes()).hexdigest();pre={'fixed_hashes':{str(p):h},'weight':str(p),'weight_sha256':h};check_files(pre);p.write_bytes(b'changed')
 with pytest.raises(AssertionError):check_files(pre)
