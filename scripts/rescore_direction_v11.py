"""Replay all 40 existing candidates with v10b/v11; never generate or select videos."""
import argparse,hashlib,json,os,time
from pathlib import Path
from dataclasses import asdict,replace
from driveloop.schema import Generation
from driveloop.acceptance import joint_gate
from driveloop.perception_v10 import ManeuverViewRestrictedSuperclassEvaluator
from driveloop.perception_v11 import ContinuousTrackPerceptionEvaluator


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(path,data):path.write_text(json.dumps(data,indent=2,ensure_ascii=False,default=str)+'\n')

def inventory(root):
    jobs=[]
    groups=[('rain_pilot','revision_20261005_M2_pilot'),('bus_before_guard','revision_20261005_M2_expansion/candidate28'),
            ('bus_after_guard','revision_motion_guard_bus_20261005_214437_48463')]
    for group,folder in groups:
        run=root/'outputs'/folder;baseline=json.loads((run/'baseline/generation.json').read_text())['artifacts']['video']
        for arm in ('direct','fixed','full'):
            path=run/arm/'result.json';r=json.loads(path.read_text())
            assert len(r['attempt_history'])==4
            for a in r['attempt_history']:
                e=a['evaluation'];score=e['metrics'].get('S_perc',e['score'])
                jobs.append({'group':group,'arm':arm,'iteration':a['iteration'],'generation':a['generation'],
                    'baseline_video':baseline,'original_score':score,'original_direction':e['metrics'].get('maneuver_direction_consistent'),
                    'original_selected':a['iteration']==r['returned_generation']['iteration'],'source_json':str(path)})
    run=root/'outputs/bus_height_pair_20261006_010720_56265';p=json.loads((run/'protocol.json').read_text())
    for a in json.loads((run/'summary.json').read_text())['cases']:
        jobs.append({'group':'bus_height_pair','arm':a['mode'],'iteration':a['seed_offset'],'generation':a['generation'],
            'baseline_video':p['baseline_video'],'original_score':a['S_perc'],'original_direction':a['direction'],
            'original_selected':None,'source_json':str(run/'summary.json')})
    assert len(jobs)==40
    return jobs


class CachedDetector:
    def __init__(self,detector):self.detector=detector;self.cache={}
    def detect(self,frame,frame_index):
        key=hashlib.sha256(frame.tobytes()).hexdigest()
        if key not in self.cache:self.cache[key]=self.detector.detect(frame,0)
        return [replace(d,frame_index=frame_index) for d in self.cache[key]]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    root=Path.cwd();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    jobs=inventory(root);video_paths=sorted({j['generation']['artifacts']['video'] for j in jobs}|{j['baseline_video'] for j in jobs})
    files=sorted((root/'driveloop').rglob('*.py'))+[Path(__file__).resolve()]
    protocol={'version':'v11_direction_rescore_1','check_only':args.check_only,'candidate_count':len(jobs),
        'source_code_sha256':{str(p.relative_to(root)):sha(p) for p in files},
        'video_sha256':{p:sha(p) for p in video_paths},
        'source_json_sha256':{p:sha(p) for p in sorted({j['source_json'] for j in jobs})},
        'yolo_weights_sha256':sha(root/'yolov8x.pt'),
        'direction_rule':{'iou_threshold':0.3,'min_consecutive_frames':3,'max_gap':0,
                          'unique_candidate_track':True,'ambiguous_match':'unmeasured','supported_view':'cam_front'},
        'unchanged':{'configured_confidence':0.2,'baseline_iou':0.5,'perception_gate':0.3,'crop':'bottom 256, tile width 448',
                     'S_perc_formula':'v10b unchanged','detector_prediction_defaults':'unchanged from v10b'},
        'historical_results_overwritten':False,'output_selection_changed':False,
        'generation_performed':False,'claim':'Retrospective geometric direction diagnostic; no human identity or lane-entry ground truth.'}
    write(out/'protocol.json',protocol)
    print('PREFLIGHT_PASS',len(jobs),'candidates;',len(video_paths),'video files;',len(set(protocol['video_sha256'].values())),'unique video hashes',flush=True)
    if args.check_only:return
    from driveloop.perception_video import UltralyticsYOLODetector
    detector=CachedDetector(UltralyticsYOLODetector(str(root/'yolov8x.pt'),confidence_threshold=0.2))
    old=ManeuverViewRestrictedSuperclassEvaluator(detector=detector,confidence_threshold=0.2)
    new=ContinuousTrackPerceptionEvaluator(detector=detector,confidence_threshold=0.2)
    rows=[]
    for index,job in enumerate(jobs):
        assert all(sha(root/p)==v for p,v in protocol['source_code_sha256'].items()),'Code changed during rescore'
        old.baseline_video=new.baseline_video=job['baseline_video']
        generation=Generation(**job['generation'])
        prior=old.evaluate(generation);current=new.evaluate(generation)
        direction=prior.metrics.get('maneuver_direction_consistent')
        if abs(prior.score-job['original_score'])>1e-6 or direction!=job['original_direction']:
            write(out/'replay_mismatch.json',{'job':{k:v for k,v in job.items() if k!='generation'},'replayed':asdict(prior)})
            raise RuntimeError('Historical evaluator did not reproduce; stop before interpreting v11 differences')
        assert current.score==prior.score,'Direction update must not change S_perc'
        row={k:v for k,v in job.items() if k!='generation'}
        row.update(old_gate=joint_gate(prior.metrics,.3,raw_score=prior.score),new_gate=joint_gate(current.metrics,.3,raw_score=current.score),
                   old_evaluation=asdict(prior),new_evaluation=asdict(current),direction_audit=new.last_direction_audit,
                   video=generation.artifacts['video'],video_sha256=protocol['video_sha256'][generation.artifacts['video']])
        rows.append(row)
        # Preserve per-view boxes for later independent review without re-running detection.
        write(out/f'candidate_{index:02}.json',{'comparison':row,'residual_detections_by_view':old._view_detection_frames})
        write(out/'partial_results.json',rows)
        print('RESCORED',index+1,'/',len(jobs),job['group'],job['arm'],job['iteration'],row['old_gate'],'->',row['new_gate'],flush=True)
    detector.detector.release_gpu()
    write(out/'summary.json',{'protocol':str(out/'protocol.json'),'complete':True,'results':rows,
        'score_changes':0,'changed_gate_count':sum(r['old_gate']!=r['new_gate'] for r in rows),
        'old_pass_count':sum(r['old_gate']=='PASS' for r in rows),'new_pass_count':sum(r['new_gate']=='PASS' for r in rows)})
    print('COMPLETE',out,flush=True)

if __name__=='__main__':main()
