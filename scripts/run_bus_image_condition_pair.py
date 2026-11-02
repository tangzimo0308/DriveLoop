"""Eight-render image-latent diagnostic with seed/scale-matched no-injection supports."""
import argparse,copy,json,os,time
from dataclasses import asdict,replace
from pathlib import Path
from driveloop.backends.drivedreamer2 import DriveDreamer2Backend
from driveloop.schema import DriveLoopRequest
from driveloop.ground_alignment import MODE
from driveloop.acceptance import joint_gate
from scripts.run_bus_height_pair import sha,save


class ProbeBackend(DriveDreamer2Backend):
    def __init__(self,*args,actor_injection=True,**kwargs):
        super().__init__(*args,**kwargs);self.actor_injection=actor_injection
    def _build_override_json(self,*args,**kwargs):
        result=super()._build_override_json(*args,**kwargs)
        assert result['boxes3d']['append']==[] and result['boxes3d']['per_frame_append']==[]
        if not self.actor_injection:
            result=copy.deepcopy(result);result['boxes3d']['per_frame_append_ego']=[]
        result['audit']['image_condition_probe_actor_injection']=self.actor_injection
        return result


def make_backend(root,out,manifest,actor):
    return ProbeBackend(project_root=root,baseline_dataset_dir=manifest['dataset'],artifact_dir=out,
        source_candidate_id=manifest['window']['source_candidate_id'],scene_token=manifest['window']['scene_token'],
        instance_token=manifest['window']['instance_token'],source_identity_summary_path=manifest['identity'],
        actor_injection=actor,timeout_seconds=3600)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    root=Path.cwd();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    old=root/'outputs/revision_motion_guard_bus_20261005_214437_48463'
    manifest=json.loads((old/'protocol.json').read_text())
    original=json.loads((old/'full/result.json').read_text())['attempt_history'][0];m=original['generation']['metadata']
    previous_height=root/'outputs/bus_height_pair_20261006_010720_56265'
    controls={a['seed_offset']:a for a in json.loads((previous_height/'summary.json').read_text())['cases'] if a['mode']==MODE}
    weight=Path('/data/models/dd2/pretrained_models/drivedreamer2_img_cond/pytorch_gligen_weights.bin')
    for name in list(os.environ):
        if name.startswith('DRIVELOOP_'):del os.environ[name]
    os.environ.update(DRIVELOOP_EGO_INJECTION='1',DRIVELOOP_EGO_GROUND_ALIGNMENT=MODE,
        DRIVELOOP_DD2_NUM_INF_STEPS='50',DRIVELOOP_DD2_SEED_BANK='0',DRIVELOOP_DD2_WEIGHT_PATH=str(weight),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    assert sha(weight)==manifest['dd2_weight_sha256']
    assert sha(root/'yolov8x.pt')==manifest['detector_weights_sha256']
    assert sha(Path(manifest['dataset'])/'labels/data.pkl')==manifest['dataset_labels_sha256']
    assert sha(manifest['identity'])==manifest['identity_sha256']
    condition=copy.deepcopy(original['request']['condition'])
    condition['dd2_condition']={'text_prompt':m['dd2_prompt'],'executable_condition':m['dd2_executable_condition']}
    request=DriveLoopRequest(prompt=original['generation']['prompt'],condition=condition)
    on=make_backend(root,out/'unused',manifest,True)._build_override_json(m['dd2_prompt'],m['dd2_structural_input_plan'],m['dd2_override_candidate_plan'],m['dd2_source_sample_binding'])
    stripped=copy.deepcopy(on);stripped['audit'].pop('image_condition_probe_actor_injection')
    assert stripped==controls[0]['generation']['metadata']['dd2_override_json']
    off=make_backend(root,out/'unused',manifest,False)._build_override_json(m['dd2_prompt'],m['dd2_structural_input_plan'],m['dd2_override_candidate_plan'],m['dd2_source_sample_binding'])
    off_compare=copy.deepcopy(off);off_compare['boxes3d']['per_frame_append_ego']=on['boxes3d']['per_frame_append_ego'];off_compare['audit']['image_condition_probe_actor_injection']=True
    assert off_compare==on,'Matched baseline changed more than actor boxes'
    files=list((root/'driveloop').rglob('*.py'))+[Path(__file__).resolve(),root/'scripts/run_bus_height_pair.py']
    for folder in ('dreamer-train/projects/DriveDreamer2','dreamer-models/dreamer_models/pipelines/drivedreamer2'):
        files.extend((root/folder).rglob('*.py'))
    code={str(p.relative_to(root)):sha(p) for p in sorted(set(files))}
    protocol={'experiment':'image_condition_latent_pair_v1','check_only':args.check_only,'seed_offsets':[0,1],
        'image_condition_scales':[1.,0.],'actor_injection':[False,True],'render_count':8,'inference_steps':50,
        'request':asdict(request),'source_protocol':manifest,'code_sha256':code,
        'historical_control_video_sha256':{str(seed):c['video_sha256'] for seed,c in controls.items()},
        'preflight_actor_override':on,'preflight_support_override':off,'evaluator':'ContinuousTrackPerceptionEvaluator v11',
        'fixed_confidence':0.2,'joint_threshold':0.3,'matched_baseline':'same seed, scale, prompt, source, map, with injected actor boxes removed',
        'claim_boundary':'Ablation of the VAE image latent under the same img_cond weights, potentially out of distribution; not a separately trained wo_img model or proof of semantic success.'}
    save(out/'protocol.json',protocol);print('PREFLIGHT_PASS: same height-corrected actor plan; matched supports; 8 renders declared.',flush=True)
    if args.check_only:return
    from driveloop.perception_v11 import ContinuousTrackPerceptionEvaluator
    from driveloop.perception_video import UltralyticsYOLODetector
    class AuditEvaluator(ContinuousTrackPerceptionEvaluator):
        def _subtract_baseline(self,detections,baseline_dets):
            kept,removed=super()._subtract_baseline(detections,baseline_dets)
            self.subtraction_records.append({'raw':[asdict(d) for d in detections],
                'baseline':[asdict(d) for d in baseline_dets],'residual':[asdict(d) for d in kept]})
            return kept,removed
    rows=[]
    for seed in (0,1):
        signatures=[];runtime_inputs={}
        for scale in (1.,0.):
            pair={}
            for actor in (False,True):
                assert all(sha(root/p)==v for p,v in code.items()),'Code changed during experiment'
                case=out/f'seed{seed}_scale{int(scale)}_{"actor" if actor else "support"}';case.mkdir()
                os.environ['DRIVELOOP_DD2_IMAGE_CONDITION_SCALE']=str(scale)
                os.environ['DRIVELOOP_DD2_IMAGE_CONDITION_AUDIT_PATH']=str(case/'image_latent_audit.json')
                start=time.monotonic();g=make_backend(root,case/'artifacts',manifest,actor).generate(request,iteration=seed)
                g=replace(g,metadata={**g.metadata,**{k:m[k] for k in ('scene_specification','long_tail_condition_plan','dd2_condition','control_coverage')}})
                audit=json.loads((case/'image_latent_audit.json').read_text())
                assert audit['scale']==scale
                if scale==0:assert audit['image_latent_after']['nonzero']==0
                else:assert audit['image_latent_after']==audit['image_latent_before']
                signatures.append(audit)
                runtime_inputs[(scale,actor)]=g.metadata['dd2_runtime_input_audit']
                digest=sha(g.artifacts['video'])
                if actor and scale==1:
                    assert digest==protocol['historical_control_video_sha256'][str(seed)],'Control video did not reproduce; stop before interpreting differences'
                row={'seed':seed,'image_scale':scale,'actor_injection':actor,'generation':asdict(g),
                     'video_sha256':digest,'seconds':time.monotonic()-start,'latent_audit':audit}
                save(case/'generation.json',row);rows.append(row);pair[actor]=g
                print('GENERATED',seed,scale,actor,flush=True)
            ev=AuditEvaluator(detector=UltralyticsYOLODetector(str(root/'yolov8x.pt'),confidence_threshold=.2),
                confidence_threshold=.2,baseline_video=pair[False].artifacts['video'])
            ev.subtraction_records=[]
            evaluation=ev.evaluate(pair[True]);ev.detector.release_gpu()
            rows[-1].update(evaluation=asdict(evaluation),direction_audit=ev.last_direction_audit,
                gate=joint_gate(evaluation.metrics,.3,raw_score=evaluation.score),support_video=pair[False].artifacts['video'],
                detection_audit={'view_order':ev._views_to_evaluate(pair[True].metadata),'frames_per_view':8,
                                 'subtraction_records':ev.subtraction_records,'residual_by_view':ev._view_detection_frames})
            save(out/f'seed{seed}_scale{int(scale)}_evaluation.json',rows[-1])
            print('EVALUATED',seed,scale,evaluation.score,rows[-1]['gate'],flush=True)
        for key in ('initial_noise','image_latent_before'):
            assert len({a[key]['sha256'] for a in signatures})==1, f'{key} changed inside seed {seed}; comparison confounded'
        for key in ('prompt_embed','img_cond','grounding_downsampler_input'):
            assert len({a[key]['sha256'] for a in runtime_inputs.values()})==1, f'{key} changed inside seed {seed}'
        for actor in (False,True):
            assert runtime_inputs[(1.,actor)]['box_downsampler_input']['sha256']==runtime_inputs[(0.,actor)]['box_downsampler_input']['sha256'], 'Box input changed across image scales'
        save(out/'partial_results.json',rows)
    save(out/'summary.json',{'complete':True,'protocol':str(out/'protocol.json'),'cases':rows,'selection_changed':False})
    print('EXPERIMENT_COMPLETE',out,flush=True)

if __name__=='__main__':main()
