"""One-seed 2x2 prompt/category-channel diagnostic; not a DriveLoop benchmark."""
import argparse
import copy
import json
import os
import re
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from driveloop.backends.drivedreamer2 import DriveDreamer2Backend
from scripts.run_screened_bus_pair import ROOT, SOURCE, context, setup_env, make_backend
from scripts.run_bus_height_pair import sha, save

EXPERIMENT='candidate1318_category_cross_v1'
PREFLIGHT=ROOT/'outputs/revision_20261006_category_cross_preflight/preflight.json'
FACTORS=[('bus','bus'),('bus','truck'),('truck','bus'),('truck','truck')]
SEED=0


def case_id(prompt_category,box_category):
    return f'prompt_{prompt_category}__box_{box_category}'


def make_variants(manifest,entries,metadata,request):
    original_prompt=metadata['dd2_prompt']
    assert len(re.findall(r'\bbus\b',original_prompt))==1,original_prompt
    base=make_backend(manifest,entries,SOURCE/'unused_category_cross',True)._build_override_json(
        original_prompt,metadata['dd2_structural_input_plan'],metadata['dd2_override_candidate_plan'],manifest['binding'])
    variants=[]
    for text_category,box_category in FACTORS:
        prompt=re.sub(r'\bbus\b',text_category,original_prompt)
        planned=copy.deepcopy(entries)
        for entry in planned:
            assert entry['category']=='bus'
            entry['category']=box_category
        override=copy.deepcopy(base)
        override['boxes3d']['per_frame_append_ego']=planned
        override['audit']['experimental_executed_plan']={
            'protocol':EXPERIMENT,'prompt_category':text_category,'box_category':box_category,
            'geometry_source':str(SOURCE/'selected_entries.json'),
            'claim':'Frozen geometry, only prompt noun and injected box category are experimental factors; upstream planner metadata is not the executed plan.'}
        cond=copy.deepcopy(request.condition)
        cond['dd2_condition']['text_prompt']=prompt
        req=replace(request,prompt=prompt,condition=cond)
        variants.append({'case_id':case_id(text_category,box_category),'prompt_category':text_category,
                         'box_category':box_category,'request':asdict(req),'override':override})
    return variants


class CategoryBackend(DriveDreamer2Backend):
    def __init__(self,*args,experiment_override,**kwargs):
        self.experiment_override=experiment_override
        super().__init__(*args,**kwargs)
    def _build_override_json(self,*args,**kwargs):
        # Intentionally freeze all executable conditioning other than these two factors.
        return copy.deepcopy(self.experiment_override)


def backend(manifest,variant,out):
    return CategoryBackend(project_root=ROOT,baseline_dataset_dir=manifest['dataset'],artifact_dir=out,
        source_candidate_id=manifest['source_candidate_id'],scene_token=manifest['scene_token'],
        instance_token=manifest['instance_token'],source_identity_summary_path=manifest['identity'],
        python_executable=sys.executable,timeout_seconds=3600,experiment_override=variant['override'])


def pinned_files(manifest):
    files=list((ROOT/'driveloop').rglob('*.py'))
    for folder in ('dreamer-train/projects/DriveDreamer2','dreamer-models/dreamer_models','dreamer-datasets/dreamer_datasets'):
        files.extend((ROOT/folder).rglob('*.py'))
    files.extend([Path(__file__).resolve(),ROOT/'scripts/preflight_category_cross.py',ROOT/'scripts/run_screened_bus_pair.py',
        ROOT/'scripts/run_bus_height_pair.py',ROOT/'ENV.py',SOURCE/'selected_entries.json',SOURCE/'candidate1318/manifest.json',
        SOURCE/'runtime_preflight.json',Path(manifest['identity']),PREFLIGHT,
        Path('/data/projects/DriveLoop/clip_text_transform_after_pool_panoramic.pkl')])
    files.extend(Path(manifest['dataset'])/part/'data.pkl' for part in ('labels',))
    files.extend(Path(manifest['dataset'])/part/'data.mdb' for part in ('images','hdmaps'))
    return {str(p):sha(p) for p in sorted(set(files))}


def check_runtime_pairs(rows):
    for row in rows:
        a=row['generation']['metadata']['dd2_runtime_input_audit'];latent=row['latent_audit']
        assert a['prompt_override']==row['request']['condition']['dd2_condition']['text_prompt']
        assert latent['scale']==1 and latent['image_latent_before']==latent['image_latent_after']
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            for key in ('initial_noise','image_latent_before'):
                assert a['latent_audit'][key]['sha256']==b['latent_audit'][key]['sha256'],key
            aa=a['generation']['metadata']['dd2_runtime_input_audit'];bb=b['generation']['metadata']['dd2_runtime_input_audit']
            for key in ('img_cond','grounding_downsampler_input'):
                assert aa[key]['sha256']==bb[key]['sha256'],key
            for factor,key in (('prompt_category','prompt_embed'),('box_category','box_downsampler_input')):
                assert (a[factor]==b[factor])==(aa[key]['sha256']==bb[key]['sha256']),(factor,key)


def make_contact_sheet(rows,out):
    import cv2
    from PIL import Image,ImageDraw
    sheet=Image.new('RGB',(448*8,292*4),'white');draw=ImageDraw.Draw(sheet)
    for i,row in enumerate(rows):
        video=Path(row['generation']['artifacts']['video']);assert sha(video)==row['video_sha256']
        cap=cv2.VideoCapture(str(video));count=0
        try:
            while True:
                ok,frame=cap.read()
                if not ok:break
                assert frame.shape[1]==448*6 and frame.shape[0]>=256 and count<8,frame.shape
                crop=Image.fromarray(cv2.cvtColor(frame[-256:,448:896],cv2.COLOR_BGR2RGB))
                sheet.paste(crop,(count*448,i*292+36));count+=1
        finally:cap.release()
        assert count==8,count
        draw.text((8,i*292+10),row['case_id']+' | seed offset 0 | frames 0-7 left to right',fill='black')
    sheet.save(out/'front_view_comparison.jpg',quality=95)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--check-only',action='store_true');args=p.parse_args()
    setup_env();manifest,entries,m,request=context();variants=make_variants(manifest,entries,m,request)
    pre=json.loads(PREFLIGHT.read_text());assert pre['passed'] and pre['variants']==variants
    for path,digest in pre['fixed_hashes'].items():assert sha(path)==digest,path
    original=json.loads((ROOT/'outputs/revision_motion_guard_bus_20261005_214437_48463/protocol.json').read_text())
    weight_path=Path(os.environ['DRIVELOOP_DD2_WEIGHT_PATH']);weight_sha=sha(weight_path)
    assert weight_sha==original['dd2_weight_sha256']
    fixed=pinned_files(manifest)
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    protocol={'experiment':EXPERIMENT,'render_count':4,'seed_offset':SEED,'seed_bank':0,'steps':50,'image_condition_scale':1,
        'manifest':manifest,'variants':variants,'fixed_hashes':fixed,'weight_path':str(weight_path),'weight_sha256':weight_sha,
        'runtime_python':sys.executable,'preflight':str(PREFLIGHT),'check_only':args.check_only,
        'claim_boundary':'One source and one seed exploratory diagnostic; no performance, category accuracy, maneuver success or safety claim. Inconsistent text/box conditions are deliberate. No post-hoc best case selection.'}
    save(out/'protocol.json',protocol)
    print('PREFLIGHT_PASS: 2 prompts x 2 box categories; 4 renders; frozen geometry; seed offset 0.',flush=True)
    if args.check_only:return
    from driveloop.schema import DriveLoopRequest
    rows=[]
    for variant in variants:
        assert all(sha(k)==v for k,v in fixed.items()),'Pinned code or inputs changed'
        assert sha(weight_path)==weight_sha,'Weights changed'
        case=out/variant['case_id'];case.mkdir();os.environ['DRIVELOOP_DD2_IMAGE_CONDITION_AUDIT_PATH']=str(case/'latent_audit.json')
        req=DriveLoopRequest(**variant['request']);start=time.monotonic()
        g=backend(manifest,variant,case/'artifacts').generate(req,iteration=SEED)
        assert g.metadata['dd2_source_sample_binding']['ready']
        assert g.metadata['dd2_source_sample_binding']['selector']['scene_token']==manifest['scene_token']
        assert g.metadata['dd2_override_json']==variant['override']
        g=replace(g,metadata={**g.metadata,'experimental_executed_plan':variant['override']['audit']['experimental_executed_plan']})
        row={**{k:variant[k] for k in ('case_id','prompt_category','box_category','request')},'seed_offset':SEED,
             'generation':asdict(g),'seconds':time.monotonic()-start,'video_sha256':sha(g.artifacts['video']),
             'latent_audit':json.loads((case/'latent_audit.json').read_text())}
        rows.append(row);save(case/'generation.json',row);save(out/'partial_results.json',rows)
        check_runtime_pairs(rows)
        print('GENERATED',variant['case_id'],round(row['seconds'],2),'seconds; factor checks passed',flush=True)
    assert all(sha(k)==v for k,v in fixed.items()),'Pinned code or inputs changed'
    make_contact_sheet(rows,out)
    save(out/'summary.json',{'complete':True,'cases':rows,'runtime_pair_checks_passed':True,'protocol':str(out/'protocol.json'),
         'claim':'No success scores; inspect all videos for bus/truck appearance. Detector labels alone are not ground truth.'})
    print('EXPERIMENT_COMPLETE',out,flush=True)

if __name__=='__main__':main()
