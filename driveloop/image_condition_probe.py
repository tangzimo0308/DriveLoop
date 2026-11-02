"""Opt-in image-latent intervention; never skip VAE sampling or change shape."""
import math
import hashlib
import json
from pathlib import Path


def validate_scale(value):
    value=float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('image_condition_scale must be finite and within [0, 1]')
    return value


def apply_image_condition_scale(condition,scale):
    scale=validate_scale(scale)
    if scale==1.0:return condition
    return condition*scale


def write_probe_audit(path,scale,noise,before,after):
    def signature(tensor):
        a=tensor.detach().cpu().contiguous().numpy()
        return {'shape':list(a.shape),'dtype':str(a.dtype),'sha256':hashlib.sha256(a.tobytes()).hexdigest(),
                'nonzero':int((a!=0).sum())}
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps({'scale':float(scale),'initial_noise':signature(noise),
        'image_latent_before':signature(before),'image_latent_after':signature(after),
        'scope':'Only VAE image-conditioning latent channel scaled; VAE sampling retained. No claim of equivalence to a separately trained unconditioned model.'},indent=2)+'\n')
