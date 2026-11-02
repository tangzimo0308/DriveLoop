"""Four-render candidate1318 diagnostic; selected entirely by source geometry."""
import argparse,copy,json,os,time
from dataclasses import asdict,replace
from pathlib import Path
from driveloop.backends.drivedreamer2 import DriveDreamer2Backend
from driveloop.schema import DriveLoopRequest
from driveloop.acceptance import joint_gate
from driveloop.plan_geometry import require_geometry_ready
from scripts.run_bus_height_pair import sha,save
ROOT=Path('/home/lesliton/projects/DriveLoop')
SOURCE=ROOT/'outputs/revision_20261006_source_screen_expanded'

class ScreenedBackend(DriveDreamer2Backend):
    def __init__(self,*args,planned_entries,actor_injection,**kwargs):
        self.planned_entries=planned_entries;self.actor_injection=actor_injection
        super().__init__(*args,**kwargs)
    def _build_override_json(self,*args,**kwargs):
        override=super()._build_override_json(*args,**kwargs)
        override['boxes3d']['mode']='append_and_per_frame_append_ego'
        override['boxes3d']['append']=[];override['boxes3d']['per_frame_append']=[]
        override['boxes3d']['per_frame_append_ego']=copy.deepcopy(self.planned_entries) if self.actor_injection else []
        override['audit']['experimental_executed_plan']={'protocol':'candidate1318_map_lane_diagnostic_v1','actor_injection':self.actor_injection,'source_entries':str(SOURCE/'selected_entries.json'),'claim':'Overrides upstream planned boxes with frozen map-screened entries; not a full DriveLoop method comparison.'}
        return override

def make_backend(manifest,entries,out,actor):
    return ScreenedBackend(project_root=ROOT,baseline_dataset_dir=manifest['dataset'],artifact_dir=out,
      source_candidate_id=manifest['source_candidate_id'],scene_token=manifest['scene_token'],instance_token=manifest['instance_token'],
      source_identity_summary_path=manifest['identity'],planned_entries=entries,actor_injection=actor,timeout_seconds=3600)

def context():
    manifest=json.loads((SOURCE/'candidate1318/manifest.json').read_text());entries=json.loads((SOURCE/'selected_entries.json').read_text())
    require_geometry_ready(manifest['geometry_admission']['selected_trial']['validation'])
    assert manifest['binding']['ready'] and manifest['candidate_index']==1318 and len(entries)==8
    original=json.loads((ROOT/'outputs/revision_motion_guard_bus_20261005_214437_48463/full/result.json').read_text())['attempt_history'][0];m=original['generation']['metadata']
    condition=copy.deepcopy(original['request']['condition']);condition['dd2_condition']={'text_prompt':m['dd2_prompt'],'executable_condition':m['dd2_executable_condition']}
    # Override any legacy source selector carried by a request.
    for k in ('source_candidate_id','sample_token','scene_token','instance_token','source_identity_summary_path'):condition.pop(k,None)
    request=DriveLoopRequest(prompt=original['generation']['prompt'],condition=condition)
    return manifest,entries,m,request

def setup_env():
    for k in list(os.environ):
        if k.startswith('DRIVELOOP_'):del os.environ[k]
    os.environ.update(DRIVELOOP_EGO_INJECTION='1',DRIVELOOP_EGO_GROUND_ALIGNMENT='off',DRIVELOOP_DD2_IMAGE_CONDITION_SCALE='1',
      DRIVELOOP_DD2_NUM_INF_STEPS='50',DRIVELOOP_DD2_SEED_BANK='0',DRIVELOOP_DD2_WEIGHT_PATH='/data/models/dd2/pretrained_models/drivedreamer2_img_cond/pytorch_gligen_weights.bin',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--check-only',action='store_true');args=p.parse_args()
    setup_env();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False);manifest,entries,m,request=context()
    pre=json.loads((SOURCE/'runtime_preflight.json').read_text());assert pre['passed']
    assert pre['transform_sha256']==sha(ROOT/'dreamer-train/projects/DriveDreamer2/drivedreamer2/drivedreamer2_transforms.py')
    assert pre['entries_sha256']==sha(SOURCE/'selected_entries.json') and pre['labels_sha256']==sha(Path(manifest['dataset'])/'labels/data.pkl')
    upstream=json.loads((ROOT/'outputs/revision_motion_guard_bus_20261005_214437_48463/protocol.json').read_text())
    assert sha(os.environ['DRIVELOOP_DD2_WEIGHT_PATH'])==upstream['dd2_weight_sha256'];assert sha(ROOT/'yolov8x.pt')==upstream['detector_weights_sha256']
    files=list((ROOT/'driveloop').rglob('*.py'))+[Path(__file__).resolve(),ROOT/'scripts/run_bus_height_pair.py']
    for sub in ('dreamer-train/projects/DriveDreamer2','dreamer-models/dreamer_models/pipelines/drivedreamer2'):files.extend((ROOT/sub).rglob('*.py'))
    fixed={str(p):sha(p) for p in files+[Path(manifest['dataset'])/'images/data.mdb',Path(manifest['dataset'])/'hdmaps/data.mdb',SOURCE/'selected_entries.json',SOURCE/'runtime_preflight.json',SOURCE/'candidate1318/manifest.json',Path(manifest['identity']),Path(manifest['dataset'])/'labels/data.pkl']}
    protocol={'experiment':'candidate1318_map_lane_diagnostic_v1','render_count':4,'seed_offsets':[0,1],'image_condition_scale':1,'steps':50,'manifest':manifest,'runtime_preflight':pre,'request':asdict(request),'fixed_hashes':fixed,'check_only':args.check_only,'evaluator':'v11','threshold':.3,'comparison':'same seed, source, map and prompt; no added actor versus frozen map-planned bus','claim_boundary':'New exploratory source and new matched baseline; not comparable as improvement over candidate28, not proof of realistic vehicle dynamics, not a full DriveLoop benchmark.'}
    save(out/'protocol.json',protocol);print('PREFLIGHT_PASS: candidate1318, four renders, two matched seeds.',flush=True)
    if args.check_only:return
    from driveloop.perception_v11 import ContinuousTrackPerceptionEvaluator
    from driveloop.perception_video import UltralyticsYOLODetector
    class AuditEvaluator(ContinuousTrackPerceptionEvaluator):
        def _subtract_baseline(self,d,b):
            kept,removed=super()._subtract_baseline(d,b);self.records.append({'raw':[asdict(x) for x in d],'baseline':[asdict(x) for x in b],'residual':[asdict(x) for x in kept]});return kept,removed
    rows=[]
    for seed in (0,1):
        pair={};audits=[]
        for actor in (False,True):
            assert all(sha(k)==v for k,v in fixed.items()),'Pinned code or inputs changed'
            case=out/f'seed{seed}_{"actor" if actor else "support"}';case.mkdir();os.environ['DRIVELOOP_DD2_IMAGE_CONDITION_AUDIT_PATH']=str(case/'latent_audit.json')
            start=time.monotonic();g=make_backend(manifest,entries,case/'artifacts',actor).generate(request,iteration=seed)
            assert g.metadata['dd2_source_sample_binding']['selector']['scene_token']==manifest['scene_token']
            assert g.metadata['dd2_override_json']['boxes3d']['per_frame_append_ego']==(entries if actor else [])
            g=replace(g,metadata={**g.metadata,**{k:m[k] for k in ('scene_specification','long_tail_condition_plan','dd2_condition','control_coverage')},'experimental_executed_plan':'candidate1318_map_lane_diagnostic_v1'})
            audit=json.loads((case/'latent_audit.json').read_text());assert audit['scale']==1 and audit['image_latent_before']==audit['image_latent_after'];audits.append(audit)
            row={'seed':seed,'actor_injection':actor,'generation':asdict(g),'seconds':time.monotonic()-start,'video_sha256':sha(g.artifacts['video']),'latent_audit':audit};save(case/'generation.json',row);rows.append(row);pair[actor]=g;print('GENERATED',seed,actor,flush=True)
        for key in ('initial_noise','image_latent_before'):assert audits[0][key]['sha256']==audits[1][key]['sha256']
        a=pair[False].metadata['dd2_runtime_input_audit'];b=pair[True].metadata['dd2_runtime_input_audit']
        for key in ('prompt_embed','img_cond','grounding_downsampler_input'):assert a[key]['sha256']==b[key]['sha256'],key
        assert a['box_downsampler_input']['sha256']!=b['box_downsampler_input']['sha256']
        ev=AuditEvaluator(detector=UltralyticsYOLODetector(str(ROOT/'yolov8x.pt'),confidence_threshold=.2),confidence_threshold=.2,baseline_video=pair[False].artifacts['video']);ev.records=[]
        assert ev.resolve_baseline_video(pair[True])==pair[False].artifacts['video'];evaluation=ev.evaluate(pair[True]);ev.detector.release_gpu()
        rows[-1].update(evaluation=asdict(evaluation),gate=joint_gate(evaluation.metrics,.3,raw_score=evaluation.score),direction_audit=ev.last_direction_audit,support_video=pair[False].artifacts['video'],detection_audit={'view_order':ev._views_to_evaluate(pair[True].metadata),'frames_per_view':8,'subtraction_records':ev.records,'residual_by_view':ev._view_detection_frames})
        save(out/f'seed{seed}_evaluation.json',rows[-1]);save(out/'partial_results.json',rows);print('EVALUATED',seed,evaluation.score,rows[-1]['gate'],flush=True)
    save(out/'summary.json',{'complete':True,'cases':rows,'protocol':str(out/'protocol.json')});print('EXPERIMENT_COMPLETE',out,flush=True)
if __name__=='__main__':main()
