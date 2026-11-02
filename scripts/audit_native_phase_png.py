"""Re-detect all archived PNGs, validate real measurements; no generation or online trial."""
import copy, json, pickle, time
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
import torch
from dreamer_datasets import boxes3d_utils
from driveloop.native_phase import NativePhaseRegistry, NativePhaseEvaluator, NativePhaseRefiner, digest
from driveloop.native_phase_png import PNGFrame, PNGClip, PNGPhaseMeasurement, FixedBusDetector
from driveloop.schema import Generation
from scripts.run_candidate1583_capture import index_cases, check_files
from scripts.run_native_matched_capture import validate_hooks
from scripts.run_native_advance_pair import check_pair

ROOT=Path('/home/lesliton/projects/DriveLoop')
RUN=ROOT/'outputs/candidate1583_capture_20261009_084734_12216'
PRIOR=ROOT/'outputs/revision_20261009_candidate1583_analysis'
OUT=ROOT/'outputs/revision_20261013_native_phase_png_audit'

def save(name,value): (OUT/name).write_text(json.dumps(value,indent=2)+'\n')

def main():
    OUT.mkdir(exist_ok=False); started=time.monotonic(); torch.set_num_threads(4)
    registry=NativePhaseRegistry(ROOT/'outputs/revision_20261008_candidate1583_inputs/preflight.json','997384cdb119b1679abc84e48276da54f38e3141d297271e50e192bd499d9af2')
    summary_path=RUN/'summary.json'; summary_hash=digest(summary_path)
    summary=json.loads(summary_path.read_text()); protocol=json.loads((RUN/'protocol.json').read_text())
    prior_protocol=json.loads((PRIOR/'protocol.json').read_text()); prior=json.loads((PRIOR/'results.json').read_text())
    assert summary_hash==prior_protocol['run_summary_sha256']
    assert digest(RUN/'protocol.json')==prior_protocol['capture_protocol_sha256']
    assert summary['complete'] and summary['all_captures_validated'] and summary['paired_input_checks_passed']
    assert protocol['preflight_sha256']==registry.fingerprint
    rows=index_cases(summary['cases']); check_files(protocol['preflight'])
    case=registry.case; plans={}; clips={}
    core=json.loads((ROOT/'outputs/revision_20261009_closed_loop_native_audit/results.json').read_text())['files_sha256']
    core[str(ROOT/'driveloop/native_phase.py')]=digest(ROOT/'driveloop/native_phase.py')
    for path,h in core.items():assert digest(path)==h,path
    for arm in ('original','delayed','advanced'):
        records=pickle.loads((Path(case['inputs'][arm]['dataset'])/'labels/data.pkl').read_bytes()); boxes=[]
        for i in range(8):
            r=records[24+i*3]; j=r['instance_tokens'].index(registry.instance)
            cs=boxes3d_utils.boxes3d_to_corners3d(r['boxes3d'][j:j+1],rot_axis=1);assert cs[...,2].min()>.5
            pts=boxes3d_utils.corners3d_to_corners2d(cs,cam_intrinsic=r['calib']['cam_intrinsic'])[0].astype(np.float64)*np.array([448/1600,256/900])
            b=np.r_[pts.min(0),pts.max(0)].tolist(); assert np.allclose(b,prior['plans'][i][arm],atol=1e-9,rtol=0)
            boxes.append(tuple(b))
        plans[arm]=boxes
    for (seed,arm),row in rows.items():
        meta=row['generation']['metadata'];binding=meta['dd2_source_sample_binding']
        assert row['scene_token']==registry.scene and binding['ready']
        assert binding['selector']['scene_token']==registry.scene and binding['selector']['instance_token']==registry.instance
        assert row['generation']['iteration']==seed and not meta['dd2_audit_only']
        assert meta['dd2_prompt'] is None and not meta['dd2_override_json']['available']
        for key in ('prompt_embed','img_cond','grounding_downsampler_input','box_downsampler_input'):
            assert meta['dd2_runtime_input_audit'][key]['sha256']==case['runtime_audits'][arm][key]['sha256']
        if arm!='original':check_pair(rows[(seed,'original')],row)
        assert digest(row['generation']['artifacts']['video'])==row['video_sha256']
        h=validate_hooks(Path(row['capture_dir']))
        fs=tuple(PNGFrame(i,str(Path(row['capture_dir'])/f'generated_front/frame{i}.png'),h['files'][f'generated_front/frame{i}.png'],plans[arm][i]) for i in range(8))
        clips[(seed,arm)]=PNGClip(registry.scene,registry.instance,registry.fingerprint,seed,arm,row['latent_audit']['initial_noise']['sha256'],fs,str(summary_path),summary_hash)
    detector_path='/data/models/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth'
    save('protocol.json',{'generation_run':False,'mode':'archived_PNG_real_detection_callback_audit','source_summary_sha256':summary_hash,
        'source_protocol_sha256':digest(RUN/'protocol.json'),'prior_analysis_sha256':digest(PRIOR/'results.json'),
        'source_run':str(RUN),'core_hashes':core,'module_sha256':digest(ROOT/'driveloop/native_phase_png.py'),'script_sha256':digest(__file__),
        'primary':'lossless PNG only','confidence':.5,'unique_bus_iou':.3,'late_indices':[4,5,6,7],
        'test_goal_px':0,'test_tolerance_px':1,'test_goal_scope':'Evaluator plumbing only; neither is a paper success criterion.',
        'limits':'Development scene already inspected; no online video generation, no new efficacy evidence, no thresholds tuned.'})
    detector=FixedBusDetector(detector_path)
    def current_for_generation(g):
        arm=g.metadata['native_phase_contract']['arm'];row=rows[(g.iteration,arm)]
        # The only injected metadata is the explicit test contract; all original
        # generation fields and paths must still be exactly the archived record.
        old=row['generation']
        assert g.artifacts==old['artifacts'] and g.prompt==old['prompt']
        assert {k:v for k,v in g.metadata.items() if k!='native_phase_contract'}==old['metadata']
        return clips[(g.iteration,arm)]
    callback=PNGPhaseMeasurement(current_for_generation,lambda g:clips[(g.iteration,'original')],detector)
    evaluator=NativePhaseEvaluator(callback); refiner=NativePhaseRefiner(registry); outputs=[];max_error=0.
    for (seed,arm),row in rows.items():
        request=registry.request(0,1);q={**request.condition['native_phase'],'arm':arm};request=replace(request,condition={'native_phase':q})
        old=row['generation'];g=Generation(**{**old,'metadata':{**old['metadata'],'native_phase_contract':q}})
        evaluation=evaluator.evaluate(g); feedback=evaluation.metrics['native_phase_feedback'];provenance=feedback['provenance']
        records=provenance['current_detections'];reference=next(c for c in prior['cases'] if (c['format'],c['seed'],c['arm'])==('png',seed,arm))
        for got,expected in zip(records,reference['frames']):
            assert got['association_count']==expected['association_count']
            # Compare raw detections, not just the final aggregate.
            assert len(got['raw'])==len(expected['raw'])
            for d,e in zip(got['raw'],expected['raw']):
                assert d['label']==e['label']
                error=max(abs(d['score']-e['score']),max(abs(x-y) for x,y in zip(d['box'],e['box'])))
                max_error=max(max_error,error);assert error<=1e-5,error
        if arm=='original': expected_late=[0.0]*4
        else:
            p=next(p for p in prior['pairs'] if (p['format'],p['seed'],p['arm'])==('png',seed,arm))
            expected_late=[f['observed_dx'] for f in p['frames'][4:8]]
        assert all(abs(x-y)<=1e-5 for x,y in zip(feedback['late_deltas'],expected_late))
        assert abs(feedback['error_px']+feedback['observed_dx_px'])<=1e-9
        refinement=refiner.refine(request,evaluation)
        # Goal zero is a plumbing probe: measured edited clips should request
        # the original action, without making another generation call.
        assert refinement.condition['native_phase']['arm']=='original'
        outputs.append({'seed':seed,'arm':arm,'evaluation':asdict(evaluation),'refinement':asdict(refinement)})
        print('PNG_FEEDBACK_PASS',seed,arm,feedback['observed_dx_px'],flush=True)
    for path,h in core.items():assert digest(path)==h,path
    assert digest(summary_path)==summary_hash;registry.verify()
    save('results.json',{'passed':True,'generation_run':False,'online_closed_loop_run':False,'real_png_frames':48,
        'cases':outputs,'detector_calls':callback.detector_calls,'max_raw_detection_difference_vs_prior':max_error,
        'prior_measurements_reproduced':True,'existing_core_unchanged':True,'seconds':time.monotonic()-started})
    print('PNG_REAL_FEEDBACK_AUDIT_COMPLETE',str(OUT),flush=True)

if __name__=='__main__':main()
