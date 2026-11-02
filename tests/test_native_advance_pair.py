import copy
import pytest
from scripts.run_native_advance_pair import check_pair


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


def actual_pose_function():
    import ast
    import numpy as np
    from pathlib import Path
    path=Path(__file__).parents[1]/'scripts/prepare_native_advance.py'
    func=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='advanced_pose')
    namespace={'np':np,'token':'target','duration':1.75,'ts':np.array([0.,1.916667]),
        'centers':np.array([[0.,0.,10.],[1.916667,0.,10.]]),'headings':np.array([[0.,0.,1.],[0.,0.,1.]]),'T':lambda r:np.eye(4)}
    exec(compile(ast.Module(body=[func],type_ignores=[]),str(path),'exec'),namespace)
    record={'instance_tokens':['target'],'boxes3d':np.array([[0,0,10,10,3,2,0,0,0]],dtype=np.float32)}
    return namespace['advanced_pose'],record


def test_actual_advance_preserves_first_frame():
    import numpy as np
    f,r=actual_pose_function()
    assert np.array_equal(f(r,0.,-.125),r['boxes3d'][0])


def test_actual_advance_has_correct_sign_and_preserves_size():
    import numpy as np
    f,r=actual_pose_function();result=f(r,1.75,-.125)
    assert result[0]==pytest.approx(1.875)
    assert np.array_equal(result[3:7],r['boxes3d'][0,3:7])
    assert r['boxes3d'][0,0]==0


def test_actual_advance_refuses_extrapolation():
    f,r=actual_pose_function()
    with pytest.raises(AssertionError):f(r,1.75,-.25)


def test_actual_advance_progress_stays_monotone():
    import numpy as np
    f,r=actual_pose_function()
    positions=[f(r,float(t),-.125)[0] for t in np.linspace(0,1.75,101)]
    assert np.all(np.diff(positions)>0)
