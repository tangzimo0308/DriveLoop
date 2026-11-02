"""Two advanced-track renders paired with frozen existing native-source controls."""
import argparse,copy,json,os,shutil,time
from dataclasses import asdict
from pathlib import Path
from scripts.run_native_bus_control import ROOT,REQUEST,setup_env,make_backend,assert_native
from scripts.run_bus_height_pair import sha,save
SOURCE=ROOT/'outputs/revision_20261006_native_advance_source'
BASELINE=ROOT/'outputs/native_bus_control_20261006_193126_95686'


def context():
    m=json.loads((SOURCE/'manifest.json').read_text());assert m['geometry']['passed'] and m['selected_delay_seconds']<0
    source=copy.deepcopy(m['original_manifest']);source['dataset']=m['edited_dataset']
    baseline=json.loads((BASELINE/'summary.json').read_text());assert baseline['complete'] and baseline['native_input_checks_passed']
    protocol=json.loads((BASELINE/'protocol.json').read_text())
    # Includes all generator code, transform code, original input bytes and environment file.
    for p,h in protocol['fixed_hashes'].items():assert sha(p)==h,p
    return m,source,baseline,protocol


def check_pair(original,edited):
    a=original['generation']['metadata']['dd2_runtime_input_audit'];b=edited['generation']['metadata']['dd2_runtime_input_audit']
    for key in ('prompt_embed','img_cond','grounding_downsampler_input'):assert a[key]['sha256']==b[key]['sha256'],key
    assert a['box_downsampler_input']['sha256']!=b['box_downsampler_input']['sha256']
    for key in ('initial_noise','image_latent_before'):
        assert original['latent_audit'][key]['sha256']==edited['latent_audit'][key]['sha256'],key
    assert original['seed_offset']==edited['seed_offset']
    assert edited['latent_audit']['scale']==1 and edited['latent_audit']['image_latent_before']==edited['latent_audit']['image_latent_after']


def contact_sheet(rows,out):
    import cv2
    from PIL import Image,ImageDraw
    sheet=Image.new('RGB',(448*8,288*4),'white');draw=ImageDraw.Draw(sheet)
    for row,r in enumerate(rows):
        path=r['generation']['artifacts']['video'];assert sha(path)==r['video_sha256'];cap=cv2.VideoCapture(path);i=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            assert frame.shape==(784,2688,3) and i<8
            restored=cv2.resize(frame,(2688,772),interpolation=cv2.INTER_LINEAR)
            im=Image.fromarray(cv2.cvtColor(restored[-256:,448:896],cv2.COLOR_BGR2RGB));sheet.paste(im,(448*i,288*row+32));i+=1
        cap.release();assert i==8
        draw.text((5,288*row+8),f"seed{r['seed_offset']} | {r['arm']} | frames0-7",fill='black')
    sheet.save(out/'native_advance_comparison.jpg',quality=95)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--check-only',action='store_true');args=p.parse_args()
    setup_env();m,source,base,base_protocol=context();pre=json.loads((SOURCE/'preflight.json').read_text());assert pre['passed']
    for path,h in pre['fixed_hashes'].items():assert sha(path)==h,path
    weight=Path(os.environ['DRIVELOOP_DD2_WEIGHT_PATH']);assert sha(weight)==base_protocol['weight_sha256']
    fixed=copy.deepcopy(base_protocol['fixed_hashes']);fixed.update(pre['fixed_hashes']);fixed[str(SOURCE/'preflight.json')]=sha(SOURCE/'preflight.json')
    for r in base['cases']:
        path=r['generation']['artifacts']['video'];assert sha(path)==r['video_sha256'];fixed[path]=r['video_sha256']
    fixed[str(BASELINE/'summary.json')]=sha(BASELINE/'summary.json');fixed[str(BASELINE/'protocol.json')]=sha(BASELINE/'protocol.json')
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    protocol={'experiment':'native_bus_smooth_advance_pair_v1','new_render_count':2,'existing_control_count':2,'seed_offsets':[0,1],
        'signed_delay_seconds':m['selected_delay_seconds'],'advance_seconds':-m['selected_delay_seconds'],'prior_delay_seconds':.25,'equal_magnitude_to_prior_delay':False,'steps':50,'image_condition_scale':1,'manifest':m,'baseline_run':str(BASELINE),
        'baseline_protocol_sha256':sha(BASELINE/'protocol.json'),'weight_sha256':base_protocol['weight_sha256'],'fixed_hashes':fixed,
        'check_only':args.check_only,'request':asdict(REQUEST),'edited_runtime_audit':pre['runtime_input_audit'],
        'analysis_method':'Use all frames, both seeds, same FasterRCNN confidence0.5 and unique bus association IoU0.3 as prior analysis; compare bounding-rectangle centers and background sensitivity; retain inconclusive small changes.',
        'analysis_plan':'Compare each advanced video with its same-seed original. Expected later-frame bus position shifts right toward advanced projected boxes; magnitude differs from prior delay trial while retaining bus appearance. Inspect all frames and seeds; no cut-in or success-rate claim.',
        'claim_boundary':'Pose-conditioning sensitivity diagnostic with reused verified controls, not a physically validated maneuver or full DriveLoop benchmark.'}
    save(out/'protocol.json',protocol);print('ADVANCE_PREFLIGHT_PASS: 2 new renders, 2 frozen controls; advanced pose only.',flush=True)
    if args.check_only:return
    rows=[]
    for seed in (0,1):
        assert all(sha(p)==h for p,h in fixed.items()),'Frozen code/data/control changed'
        assert sha(weight)==base_protocol['weight_sha256']
        original=copy.deepcopy(next(r for r in base['cases'] if r['seed_offset']==seed));original['arm']='original_control_reused'
        orig_path=original['generation']['artifacts']['video'];control=out/f'seed{seed}_original';control.mkdir()
        shutil.copy2(orig_path,control/'video.mp4');original['original_video_path']=orig_path
        original['generation']['artifacts']['video']=str(control/'video.mp4');save(control/'generation.json',original);rows.append(original)
        case=out/f'seed{seed}_advanced';case.mkdir();os.environ['DRIVELOOP_DD2_IMAGE_CONDITION_AUDIT_PATH']=str(case/'latent_audit.json')
        start=time.monotonic();g=make_backend(source,case/'artifacts').generate(REQUEST,iteration=seed)
        assert_native(g,source,pre['runtime_input_audit'])
        edited={'arm':'advanced_track','seed_offset':seed,'generation':asdict(g),'video_sha256':sha(g.artifacts['video']),
            'seconds':time.monotonic()-start,'latent_audit':json.loads((case/'latent_audit.json').read_text())}
        save(case/'generation.json',edited);rows.append(edited);save(out/'partial_results.json',rows);check_pair(original,edited)
        print('GENERATED advanced seed',seed,round(edited['seconds'],2),'seconds; matched-control checks passed',flush=True)
    assert all(sha(p)==h for p,h in fixed.items())
    contact_sheet(rows,out);save(out/'summary.json',{'complete':True,'cases':rows,'paired_input_checks_passed':True,'new_renders':2,'reused_controls':2,'protocol':str(out/'protocol.json')})
    print('EXPERIMENT_COMPLETE',out,flush=True)
if __name__=='__main__':main()
