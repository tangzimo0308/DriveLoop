import copy,json
from pathlib import Path
import pytest
from tests.test_native_phase import registry
from driveloop.native_phase import digest,SIGNATURES
from driveloop.phase_input_receipt import PhaseInputReceipt
from driveloop.schema import Generation

@pytest.fixture
def packet(registry,tmp_path):
 r=registry;phase=-.0634292605404108;edited=tmp_path/'edited';edited.mkdir()
 label=edited/'label';label.write_text('exact boxes')
 manifest=tmp_path/'geometry.json'
 manifest.write_text(json.dumps({'selected_delay_seconds':phase,'geometry':{'passed':True},'instance_token':r.instance,'original_manifest':{'scene_token':r.scene},'native_dataset':r.case['inputs']['original']['dataset'],'edited_dataset':str(edited)}))
 audit=copy.deepcopy(r.case['runtime_audits']['original']);audit['box_downsampler_input']['sha256']='new_boxes'
 data={'schema':'exact_phase_input_v1','passed':True,'generation_run':False,'phase_s':phase,'registry_sha256':r.fingerprint,
       'scene_token':r.scene,'instance_token':r.instance,'candidate_index':r.candidate,'geometry_manifest':str(manifest),
       'input_manifest':{'dataset':str(edited)},'fixed_hashes':{str(manifest):digest(manifest),str(label):digest(label)},
       'geometry_passed':True,'raster_passed':True,'runtime_passed':True,'first_frames_unchanged':True,'runtime_audit':audit}
 path=tmp_path/'receipt.json';path.write_text(json.dumps(data))
 g=Generation(0,'',metadata={'dd2_audit_only':True,'dd2_source_sample_binding':{'ready':True,'dataset_dir':str(edited),'selector':{'scene_token':r.scene,'instance_token':r.instance}},'dd2_prompt':None,'dd2_override_json':{'available':False},'dd2_runtime_input_audit':audit})
 return r,phase,path,g,label

def test_exact_receipt_and_actual_input_match(packet):
 r,phase,path,g,_=packet
 assert PhaseInputReceipt(path,digest(path),phase,r).verify_runtime(g)

@pytest.mark.parametrize('change',[1e-12, .001, -.001])
def test_nearby_parameter_not_admitted(packet,change):
 r,phase,path,g,_=packet
 with pytest.raises(ValueError,match='exact phase'):PhaseInputReceipt(path,digest(path),phase+change,r)

def test_labels_tampered_after_admission(packet):
 r,phase,path,g,label=packet;a=PhaseInputReceipt(path,digest(path),phase,r);label.write_text('changed')
 with pytest.raises(ValueError,match='artifact changed'):a.verify_runtime(g)

def test_receipt_tampering(packet):
 r,phase,path,g,_=packet;h=digest(path);path.write_text('{}')
 with pytest.raises(ValueError,match='receipt changed'):PhaseInputReceipt(path,h,phase,r)

@pytest.mark.parametrize('field,value',[('instance_token','wrong'),('registry_sha256','wrong'),('phase_s',.25),('first_frames_unchanged',False),('runtime_passed',False)])
def test_invalid_packet_even_if_new_hash_supplied(packet,field,value):
 r,phase,path,g,_=packet;data=json.loads(path.read_text());data[field]=value;path.write_text(json.dumps(data))
 with pytest.raises(ValueError):PhaseInputReceipt(path,digest(path),phase,r)

@pytest.mark.parametrize('field,value',[('img_cond','wrong'),('box_downsampler_input','wrong')])
def test_actual_loader_tensor_mismatch(packet,field,value):
 r,phase,path,g,_=packet;a=PhaseInputReceipt(path,digest(path),phase,r)
 g.metadata['dd2_runtime_input_audit'][field]['sha256']=value
 with pytest.raises(ValueError,match='Runtime input differs'):a.verify_runtime(g)

def test_actual_identity_mismatch(packet):
 r,phase,path,g,_=packet;a=PhaseInputReceipt(path,digest(path),phase,r)
 g.metadata['dd2_source_sample_binding']['selector']['instance_token']='other'
 with pytest.raises(ValueError):a.verify_runtime(g)

def test_actual_dataset_mismatch(packet):
 r,phase,path,g,_=packet;a=PhaseInputReceipt(path,digest(path),phase,r)
 g.metadata['dd2_source_sample_binding']['dataset_dir']=str(path.parent/'other')
 with pytest.raises(ValueError):a.verify_runtime(g)

def test_no_video_in_input_only_mode(packet):
 r,phase,path,g,_=packet;a=PhaseInputReceipt(path,digest(path),phase,r);g.artifacts['video']='unexpected.mp4'
 with pytest.raises(ValueError):a.verify_runtime(g)
