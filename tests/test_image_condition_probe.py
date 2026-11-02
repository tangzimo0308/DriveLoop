import json
import torch
import pytest
from driveloop.image_condition_probe import apply_image_condition_scale,validate_scale,write_probe_audit


def test_scale_one_is_exact_identity():
    x=torch.randn(2,4,3,5,dtype=torch.float16)
    assert apply_image_condition_scale(x,1) is x


def test_zero_keeps_channels_and_rng_state():
    x=torch.randn(2,4,3,5,dtype=torch.float16);old=x.clone();state=torch.random.get_rng_state().clone()
    y=apply_image_condition_scale(x,0)
    assert y.shape==x.shape and y.dtype==x.dtype and torch.count_nonzero(y)==0
    assert torch.equal(x,old) and torch.equal(torch.random.get_rng_state(),state)

@pytest.mark.parametrize('scale',[-1,1.1,float('nan'),float('inf')])
def test_invalid_scales_fail(scale):
    with pytest.raises(ValueError):validate_scale(scale)

def test_half_scale_and_audit(tmp_path):
    x=torch.ones(2,4,3,5);y=apply_image_condition_scale(x,.5);assert torch.all(y==.5)
    path=tmp_path/'audit.json';write_probe_audit(path,.5,x,x,y);r=json.loads(path.read_text())
    assert r['image_latent_before']['sha256']!=r['image_latent_after']['sha256']
    assert r['initial_noise']['shape']==r['image_latent_after']['shape']

def test_actual_pipeline_branch_preserves_default_and_random_draws():
    # Execute the real image-conditioning branch with a tiny fake VAE; no model weights.
    import ast
    from pathlib import Path
    from types import SimpleNamespace
    from driveloop.image_condition_probe import write_probe_audit
    path=Path(__file__).parents[1]/'dreamer-models/dreamer_models/pipelines/drivedreamer2/pipeline_drivedreamer2.py'
    tree=ast.parse(path.read_text())
    call=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='__call__')
    block=next(n for n in call.body if isinstance(n,ast.If) and ast.unparse(n.test)=='img_cond is not None')
    module=ast.Module(body=[block],type_ignores=[])
    class VAE:
        config=SimpleNamespace(scaling_factor=.5)
        def __init__(self):self.calls=0
        def encode(self,x):
            self.calls+=1
            return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda:torch.randn(1,4,2,2)))
    results=[]
    for scale in (1.,0.):
        torch.manual_seed(31);vae=VAE();ns={'self':SimpleNamespace(vae=vae,dtype=torch.float16),
            'img_cond':torch.zeros(1,3,16,16),'latents':torch.zeros(1,8,4,2,2,dtype=torch.float16),
            'torch':torch,'do_classifier_free_guidance':True,'image_condition_scale':scale,
            'image_condition_audit_path':None,'apply_image_condition_scale':apply_image_condition_scale,
            'write_probe_audit':write_probe_audit}
        exec(compile(module,str(path),'exec'),ns)
        results.append((ns['condition_latents'],torch.random.get_rng_state().clone(),vae.calls,ns['unscaled_condition']))
    assert results[0][0].shape==results[1][0].shape==(2,8,4,2,2)
    assert torch.equal(results[0][1],results[1][1]) and results[0][2]==results[1][2]==1
    assert torch.count_nonzero(results[1][0])==0
    assert torch.equal(results[0][0][0,0],results[0][3][0])
    assert torch.equal(results[0][0][0],results[0][0][1])
