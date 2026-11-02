"""Predeclared 2-seed paired height diagnostic; not a closed-loop comparison."""
import argparse, copy, hashlib, json, os, subprocess, time
from dataclasses import asdict, replace
from pathlib import Path
from driveloop.backends.drivedreamer2 import DriveDreamer2Backend
from driveloop.schema import DriveLoopRequest
from driveloop.acceptance import joint_gate
from driveloop.ground_alignment import MODE


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str)+'\n')


def backend(root,out,manifest):
    return DriveDreamer2Backend(project_root=root,baseline_dataset_dir=manifest['dataset'],artifact_dir=out,
        source_candidate_id=manifest['window']['source_candidate_id'],scene_token=manifest['window']['scene_token'],
        instance_token=manifest['window']['instance_token'],source_identity_summary_path=manifest['identity'],timeout_seconds=3600)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    root=Path.cwd();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    previous=root/'outputs/revision_motion_guard_bus_20261005_214437_48463'
    manifest=json.loads((previous/'protocol.json').read_text())
    original=json.loads((previous/'full/result.json').read_text())['attempt_history'][0]
    m=original['generation']['metadata']
    weight=Path('/data/models/dd2/pretrained_models/drivedreamer2_img_cond/pytorch_gligen_weights.bin')
    for name in list(os.environ):
        if name.startswith('DRIVELOOP_'):del os.environ[name]
    os.environ.update(DRIVELOOP_EGO_INJECTION='1',DRIVELOOP_DD2_NUM_INF_STEPS='50',DRIVELOOP_DD2_SEED_BANK='0',
        DRIVELOOP_DD2_WEIGHT_PATH=str(weight),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    assert sha(weight)==manifest['dd2_weight_sha256']
    assert sha(root/'yolov8x.pt')==manifest['detector_weights_sha256']
    assert sha(Path(manifest['dataset'])/'labels/data.pkl')==manifest['dataset_labels_sha256']
    assert sha(manifest['identity'])==manifest['identity_sha256']
    condition=copy.deepcopy(original['request']['condition'])
    condition['dd2_condition']={'text_prompt':m['dd2_prompt'],'executable_condition':m['dd2_executable_condition']}
    request=DriveLoopRequest(prompt=original['generation']['prompt'],condition=condition)
    overrides={}
    for mode in ('off',MODE):
        os.environ['DRIVELOOP_EGO_GROUND_ALIGNMENT']=mode
        b=backend(root,out/'unused-preflight',manifest)
        overrides[mode]=b._build_override_json(m['dd2_prompt'],m['dd2_structural_input_plan'],m['dd2_override_candidate_plan'],m['dd2_source_sample_binding'])
    assert overrides['off']==m['dd2_override_json'],'Historical override replay mismatch'
    # Prove that only ego z and explanatory audit metadata changed.
    stripped=copy.deepcopy(overrides[MODE]);stripped['audit']['actor_motion_frame_mapping'].pop('ground_alignment')
    for old,new in zip(overrides['off']['boxes3d']['per_frame_append_ego'],stripped['boxes3d']['per_frame_append_ego']):
        new['ego']['center_ego'][2]=old['ego']['center_ego'][2];new.pop('ground_alignment')
    assert stripped==overrides['off'],'Unexpected change besides height'
    baseline=json.loads((previous/'baseline/generation.json').read_text())['artifacts']['video']
    files=list((root/'driveloop').rglob('*.py'))+[Path(__file__).resolve()]
    for folder in ('dreamer-train/projects/DriveDreamer2','dreamer-models/dreamer_models/pipelines/drivedreamer2'):
        files.extend((root/folder).rglob('*.py'))
    code={str(p.relative_to(root)):sha(p) for p in sorted(set(files))}
    protocol={'experiment':'paired_height_only_v1','check_only':args.check_only,'seed_offsets':[0,1],
        'arms':['off',MODE],'inference_steps':50,'request':asdict(request),'source_protocol':manifest,
        'source_code_sha256':code,'baseline_video':baseline,'baseline_sha256':sha(baseline),
        'expected_off_seed0_video_sha256':sha(previous/'full/artifacts/iteration_00.mp4'),
        'preflight_overrides':overrides,'fixed_confidence':0.2,'joint_threshold':0.3,
        'claim_boundary':'One scene, two paired seeds; source-vehicle bottom is a flat-road proxy, not surveyed terrain. Not proof of closed-loop benefit.'}
    save(out/'protocol.json',protocol)
    print('PREFLIGHT_PASS: only height changed; historical override reproduced.',flush=True)
    if args.check_only:return
    from driveloop.perception_video import UltralyticsYOLODetector
    from driveloop.perception_v10 import ManeuverViewRestrictedSuperclassEvaluator
    rows=[]
    for seed in (0,1):
        for mode in ('off',MODE):
            assert all(sha(root/p)==value for p,value in code.items()),'Source code changed during experiment'
            os.environ['DRIVELOOP_EGO_GROUND_ALIGNMENT']=mode
            case=out/f'{mode}_seed{seed}';case.mkdir()
            start=time.monotonic();generation=backend(root,case/'artifacts',manifest).generate(request,iteration=seed)
            generation=replace(generation,metadata={**generation.metadata,**{k:m[k] for k in ('scene_specification','long_tail_condition_plan','dd2_condition','control_coverage')}})
            video=generation.artifacts['video'];video_hash=sha(video)
            if mode=='off' and seed==0:
                assert video_hash==protocol['expected_off_seed0_video_sha256'],'Historical control did not reproduce; stop before comparing'
            evaluator=ManeuverViewRestrictedSuperclassEvaluator(detector=UltralyticsYOLODetector(str(root/'yolov8x.pt'),confidence_threshold=0.2),confidence_threshold=0.2,baseline_video=baseline)
            evaluation=evaluator.evaluate(generation)
            evaluator.detector.release_gpu()
            row={'mode':mode,'seed_offset':seed,'S_perc':evaluation.score,
                 'direction':evaluation.metrics.get('maneuver_direction_consistent'),
                 'gate':joint_gate(evaluation.metrics,0.3,raw_score=evaluation.score),'video_sha256':video_hash,
                 'generation':asdict(generation),'evaluation':asdict(evaluation),'wall_seconds':time.monotonic()-start}
            save(case/'result.json',row);rows.append(row)
            print('CASE_COMPLETE',mode,seed,evaluation.score,row['gate'],flush=True)
    save(out/'summary.json',{'protocol':str(out/'protocol.json'),'cases':rows})
    print('EXPERIMENT_COMPLETE',out,flush=True)


if __name__=='__main__':main()
