import copy
import pytest
from scripts.run_native_motion_pair import check_pair


def pair():
    sig=lambda x:{'sha256':x}
    base={'seed_offset':0,'latent_audit':{'scale':1,'initial_noise':sig('noise'),'image_latent_before':sig('image'),'image_latent_after':sig('image')},
     'generation':{'metadata':{'dd2_runtime_input_audit':{k:sig(k) for k in ('prompt_embed','img_cond','grounding_downsampler_input','box_downsampler_input')}}}}
    edited=copy.deepcopy(base);edited['generation']['metadata']['dd2_runtime_input_audit']['box_downsampler_input']=sig('edited-boxes')
    return base,edited


def test_correct_pair():check_pair(*pair())

@pytest.mark.parametrize('key',['prompt_embed','img_cond','grounding_downsampler_input'])
def test_unplanned_condition_change_rejected(key):
    a,b=pair();b['generation']['metadata']['dd2_runtime_input_audit'][key]['sha256']='wrong'
    with pytest.raises(AssertionError):check_pair(a,b)

@pytest.mark.parametrize('key',['initial_noise','image_latent_before'])
def test_unmatched_random_state_rejected(key):
    a,b=pair();b['latent_audit'][key]['sha256']='wrong'
    with pytest.raises(AssertionError):check_pair(a,b)


def test_unchanged_boxes_rejected():
    a,b=pair();b['generation']['metadata']['dd2_runtime_input_audit']=copy.deepcopy(a['generation']['metadata']['dd2_runtime_input_audit'])
    with pytest.raises(AssertionError):check_pair(a,b)


def test_different_seed_rejected():
    a,b=pair();b['seed_offset']=1
    with pytest.raises(AssertionError):check_pair(a,b)
