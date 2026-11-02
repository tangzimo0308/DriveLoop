"""Native bus retention control: original conditions, default prompt, two fixed seeds."""
import argparse,json,os,sys,time
from dataclasses import asdict
from pathlib import Path
from driveloop.backends.drivedreamer2 import DriveDreamer2Backend
from driveloop.schema import DriveLoopRequest
from scripts.run_bus_height_pair import sha,save
ROOT=Path('/home/lesliton/projects/DriveLoop')
SOURCE=ROOT/'outputs/revision_20261006_native_bus_source'
REQUEST=DriveLoopRequest(prompt='Native-source bus retention diagnostic; use unchanged DD2 source conditions.',condition={})


def setup_env():
    for key in list(os.environ):
        if key.startswith('DRIVELOOP_'):del os.environ[key]
    os.environ.update(DRIVELOOP_EGO_INJECTION='0',DRIVELOOP_EGO_GROUND_ALIGNMENT='off',
        DRIVELOOP_DD2_IMAGE_CONDITION_SCALE='1',DRIVELOOP_DD2_NUM_INF_STEPS='50',DRIVELOOP_DD2_SEED_BANK='0',
        DRIVELOOP_DD2_WEIGHT_PATH='/data/models/dd2/pretrained_models/drivedreamer2_img_cond/pytorch_gligen_weights.bin',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')


def make_backend(manifest,out,audit_only=False):
    return DriveDreamer2Backend(project_root=ROOT,baseline_dataset_dir=manifest['dataset'],artifact_dir=out,
        source_candidate_id=manifest['source_candidate_id'],scene_token=manifest['scene_token'],instance_token=manifest['instance_token'],
        source_identity_summary_path=manifest['identity'],python_executable=sys.executable,timeout_seconds=3600,audit_only=audit_only)


def assert_native(g,manifest,expected=None):
    meta=g.metadata;a=meta['dd2_runtime_input_audit']
    assert meta['dd2_source_sample_binding']['ready']
    assert meta['dd2_source_sample_binding']['selector']['scene_token']==manifest['scene_token']
    assert meta['dd2_override_json']['available'] is False,'Native control must not apply any overrides'
    assert meta['dd2_prompt'] is None and a['prompt_override'] is None,'Default cached DD2 prompt required'
    for key in ('prompt_embed','img_cond','grounding_downsampler_input','box_downsampler_input'):
        assert a[key]['available'] and a[key]['nonzero']>0,key
        if expected:assert a[key]['sha256']==expected[key]['sha256'],key


def fixed_inputs(manifest):
    files=list((ROOT/'driveloop').rglob('*.py'))
    for folder in ('dreamer-train/projects/DriveDreamer2','dreamer-models/dreamer_models','dreamer-datasets/dreamer_datasets'):
        files.extend((ROOT/folder).rglob('*.py'))
    files.extend([ROOT/'scripts/run_native_bus_control.py',ROOT/'scripts/preflight_native_bus_control.py',ROOT/'scripts/prepare_native_bus.py',
        ROOT/'scripts/run_bus_height_pair.py',ROOT/'ENV.py',SOURCE/'manifest.json',SOURCE/'identity.json',SOURCE/'preflight.json',
        SOURCE/'selection_protocol.json',SOURCE/'selection_ledger.json',
        Path('/data/projects/DriveLoop/clip_text_transform_after_pool_panoramic.pkl')])
    files.extend((Path(manifest['dataset'])).rglob('config.json'))
    files.extend(Path(manifest['dataset'])/part/name for part,name in [('labels','data.pkl'),('images','data.mdb'),('hdmaps','data.mdb')])
    return {str(p):sha(p) for p in sorted(set(files))}


def comparison(rows,manifest,out):
    import cv2
    from PIL import Image,ImageDraw
    from dreamer_datasets import load_dataset
    ds=load_dataset(manifest['dataset']);sheet=Image.new('RGB',(448*8,288*3),'white');draw=ImageDraw.Draw(sheet)
    # Export source and generated front views; account for encoder resize in generated composite.
    frames=[[ds[24+i*3]['image'].convert('RGB').resize((448,256)) for i in range(8)]]
    for row in rows:
        video=row['generation']['artifacts']['video'];assert sha(video)==row['video_sha256']
        cap=cv2.VideoCapture(video);views=[]
        while True:
            ok,frame=cap.read()
            if not ok:break
            assert frame.shape==(784,2688,3),frame.shape
            original=cv2.resize(frame,(2688,772),interpolation=cv2.INTER_LINEAR)
            views.append(Image.fromarray(cv2.cvtColor(original[-256:,448:896],cv2.COLOR_BGR2RGB)))
        cap.release();assert len(views)==8;frames.append(views)
    for r,views in enumerate(frames):
        draw.text((5,r*288+8),'ORIGINAL SOURCE' if r==0 else f'GENERATED native control | seed offset {rows[r-1]["seed_offset"]}',fill='black')
        for i,im in enumerate(views):sheet.paste(im,(448*i,r*288+32))
    sheet.save(out/'native_bus_comparison.jpg',quality=95)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--check-only',action='store_true');args=p.parse_args()
    setup_env();manifest=json.loads((SOURCE/'manifest.json').read_text());pre=json.loads((SOURCE/'preflight.json').read_text())
    assert pre['passed'] and pre['generation_run'] is False
    for path,digest in pre['fixed_hashes'].items():assert sha(path)==digest,path
    expected=pre['runtime_input_audit'];fixed=fixed_inputs(manifest)
    weight=Path(os.environ['DRIVELOOP_DD2_WEIGHT_PATH']);weight_sha=sha(weight)
    old=json.loads((ROOT/'outputs/category_cross_20261006_184730_89885/protocol.json').read_text())
    assert weight_sha==old['weight_sha256']
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    protocol={'experiment':'native_bus_retention_control_v1','manifest':manifest,'request':asdict(REQUEST),
        'render_count':2,'seed_offsets':[0,1],'seed_bank':0,'steps':50,'image_condition_scale':1,
        'actual_prompt':'original dataset cached DD2 embedding; no text override','weight_sha256':weight_sha,
        'fixed_hashes':fixed,'runtime_python':sys.executable,'check_only':args.check_only,'source_input_audit':expected,
        'view_export':'Generated 784px-high composite resized back to772 before cropping bottom256 rows; approximate inversion, not lossless. Raw video retained.',
        'claim':'Tests retention of an already-present visible bus; not new bus creation, cut-in success, text controllability or a full DriveLoop benchmark.'}
    save(out/'protocol.json',protocol)
    print('NATIVE_PREFLIGHT_PASS: original source conditions; no box/prompt override; two seeds.',flush=True)
    if args.check_only:return
    rows=[]
    for seed in (0,1):
        assert all(sha(k)==v for k,v in fixed.items()),'Code or source inputs changed'
        assert sha(weight)==weight_sha
        case=out/f'seed{seed}';case.mkdir();os.environ['DRIVELOOP_DD2_IMAGE_CONDITION_AUDIT_PATH']=str(case/'latent_audit.json')
        start=time.monotonic();g=make_backend(manifest,case/'artifacts').generate(REQUEST,iteration=seed)
        assert_native(g,manifest,expected)
        latent=json.loads((case/'latent_audit.json').read_text());assert latent['scale']==1 and latent['image_latent_before']==latent['image_latent_after']
        row={'seed_offset':seed,'generation':asdict(g),'video_sha256':sha(g.artifacts['video']),
             'seconds':time.monotonic()-start,'latent_audit':latent}
        rows.append(row);save(case/'generation.json',row);save(out/'partial_results.json',rows)
        print('GENERATED native seed',seed,round(row['seconds'],2),'seconds; unchanged-condition checks passed',flush=True)
    assert rows[0]['latent_audit']['initial_noise']['sha256']!=rows[1]['latent_audit']['initial_noise']['sha256']
    assert all(sha(k)==v for k,v in fixed.items())
    comparison(rows,manifest,out)
    save(out/'summary.json',{'complete':True,'cases':rows,'native_input_checks_passed':True,'protocol':str(out/'protocol.json')})
    print('EXPERIMENT_COMPLETE',out,flush=True)
if __name__=='__main__':main()
