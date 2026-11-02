"""CPU-only validation of the category-cross intervention on all 48 source records."""
import ast
import copy
import json
import os
import warnings
from pathlib import Path
import cv2
import mmengine
import numpy as np
import torch
from torchvision import transforms
from dreamer_datasets import load_dataset,boxes3d_utils
from driveloop.dd2_override import apply_dd2_override_to_sample,tensor_signature,write_override_audit
from scripts.run_category_cross import ROOT,SOURCE,PREFLIGHT,context,setup_env,make_variants
from scripts.run_bus_height_pair import sha,save


def main():
    setup_env();torch.set_num_threads(2)
    manifest,entries,m,request=context();variants=make_variants(manifest,entries,m,request)
    prior=json.loads((SOURCE/'runtime_preflight.json').read_text());assert prior['passed']
    path=ROOT/'dreamer-train/projects/DriveDreamer2/drivedreamer2/drivedreamer2_transforms.py'
    assert prior['transform_sha256']==sha(path)
    assert prior['entries_sha256']==sha(SOURCE/'selected_entries.json')
    assert prior['labels_sha256']==sha(Path(manifest['dataset'])/'labels/data.pkl')
    clean=[]
    for v in variants:
        o=copy.deepcopy(v['override']);o['audit'].pop('experimental_executed_plan')
        for e,original in zip(o['boxes3d']['per_frame_append_ego'],entries):
            assert e['category']==v['box_category'];e['category']='bus';assert e==original
        clean.append(o)
    assert all(o==clean[0] for o in clean),'Unplanned override difference'
    for v in variants:
        req=copy.deepcopy(v['request']);req['prompt']=m['dd2_prompt'];req['condition']['dd2_condition']['text_prompt']=m['dd2_prompt']
        from dataclasses import asdict
        reference=asdict(request);reference['prompt']=m['dd2_prompt']
        assert req==reference,'Unplanned request difference'
    tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DriveDreamer2_Transform')
    ns=dict(globals(),HEIGHT=900,WIDTH=1600)
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(path),'exec'),ns)
    transform=ns['DriveDreamer2_Transform'](resolution=(256,448),gd_input_name='image_hdmap',bd_input_name='image_box',is_train=False,
        embed_map_path='/data/projects/DriveLoop/clip_text_transform_after_pool_panoramic.pkl',text_embed_dim=1024,box_normal=True)
    assert transform.name_map['vehicle.bus.rigid']==11 and transform.name_map['vehicle.truck']==9
    dataset=load_dataset(manifest['dataset']);rows=[]
    for step,e in enumerate(entries):
        for cam,idx in e['source_record_indices'].items():
            raw=dataset[idx];assert raw['scene_token']==manifest['scene_token'] and raw['cam_type']==cam
            samples=[];results=[]
            for v in variants[:2]:
                sample,_=apply_dd2_override_to_sample(copy.deepcopy(raw),v['override']);samples.append(sample)
                assert len(sample['boxes3d'])==len(raw['boxes3d'])+(cam=='cam_front')
                assert np.array_equal(sample['boxes3d'][:len(raw['boxes3d'])],raw['boxes3d'])
                assert list(sample['ori_labels3d'][:len(raw['ori_labels3d'])])==list(raw['ori_labels3d'])
                if cam=='cam_front':
                    assert sample['ori_labels3d'][-1]==('vehicle.bus.rigid' if v['box_category']=='bus' else 'vehicle.truck')
                with warnings.catch_warnings():
                    warnings.simplefilter('error',RuntimeWarning);results.append(transform(sample))
            assert np.array_equal(samples[0]['boxes3d'],samples[1]['boxes3d'])
            aa=results[0]['box_downsampler_input'].numpy();bb=results[1]['box_downsampler_input'].numpy()
            changed=[c for c in range(19) if not np.array_equal(aa[c],bb[c])]
            assert changed==([9,11] if cam=='cam_front' else []),(step,cam,changed)
            for key in ('grounding_downsampler_input',):
                assert np.array_equal(results[0][key],results[1][key])
            rows.append({'step':step,'camera':cam,'record_index':idx,'changed_channels':changed,
                'bus_box_tensor':tensor_signature(aa),'truck_box_tensor':tensor_signature(bb),
                'changed_pixels':int(np.count_nonzero(aa!=bb))})
        print('INPUT_CHECK',step+1,'/8',flush=True)
    files=[path,Path(__file__).resolve(),ROOT/'scripts/run_category_cross.py',SOURCE/'selected_entries.json',SOURCE/'runtime_preflight.json',
           SOURCE/'candidate1318/manifest.json',Path(manifest['dataset'])/'labels/data.pkl',ROOT/'scripts/run_screened_bus_pair.py']
    files.extend((ROOT/'driveloop').rglob('*.py'))
    PREFLIGHT.parent.mkdir(parents=True,exist_ok=True)
    save(PREFLIGHT,{'passed':True,'gpu_generation':False,'variants':variants,'rows':rows,
        'fixed_hashes':{str(p):sha(p) for p in sorted(set(files))},
        'checks':['requests differ only by prompt noun','overrides differ only by injected category and explicit audit factors',
                  'all existing objects and all 3D geometry unchanged','only channels 9 and 11 differ in eight front frames',
                  'other 40 camera frames and all map tensors unchanged'],
        'claim':'CPU preprocessing only. Prompt embeddings, initial noise and model input equality are guarded during the user-run generation.'})
    print('CATEGORY_PREFLIGHT_PASS: 48 camera frames x 2 box categories; 4 defined requests; no video generation.',flush=True)
if __name__=='__main__':main()
