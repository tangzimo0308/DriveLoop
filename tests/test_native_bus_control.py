import copy
from types import SimpleNamespace
import pytest
from scripts.run_native_bus_control import assert_native,REQUEST
from driveloop.backends.drivedreamer2 import DriveDreamer2Backend


def native():
    audit={'prompt_override':None}
    for key in ('prompt_embed','img_cond','grounding_downsampler_input','box_downsampler_input'):
        audit[key]={'available':True,'nonzero':1,'sha256':key}
    meta={'dd2_source_sample_binding':{'ready':True,'selector':{'scene_token':'source'}},
          'dd2_override_json':{'available':False},'dd2_prompt':None,'dd2_runtime_input_audit':audit}
    return SimpleNamespace(metadata=meta),{'scene_token':'source'},copy.deepcopy(audit)


def test_empty_request_does_not_inject_any_boxes():
    assert REQUEST.condition=={}
    backend=DriveDreamer2Backend()
    result=backend._build_override_json(None,{}, {}, None)
    assert result['available'] is False


def test_valid_native_inputs():
    assert_native(*native())

@pytest.mark.parametrize('key',['prompt_embed','img_cond','grounding_downsampler_input','box_downsampler_input'])
def test_changed_native_input_is_rejected(key):
    g,m,expected=native();g.metadata['dd2_runtime_input_audit'][key]['sha256']='changed'
    with pytest.raises(AssertionError):assert_native(g,m,expected)


def test_injection_rejected():
    g,m,expected=native();g.metadata['dd2_override_json']['available']=True
    with pytest.raises(AssertionError):assert_native(g,m,expected)


def test_prompt_override_rejected():
    g,m,expected=native();g.metadata['dd2_runtime_input_audit']['prompt_override']='a bus'
    with pytest.raises(AssertionError):assert_native(g,m,expected)
