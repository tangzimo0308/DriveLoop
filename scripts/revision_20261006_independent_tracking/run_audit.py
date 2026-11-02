"""Independent detector and consecutive association, on existing videos only."""
import hashlib,json,sys,time
from pathlib import Path
from track_audit import iou,audit_tracks


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    import cv2,torch,torchvision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2,FasterRCNN_ResNet50_FPN_V2_Weights
    root=Path.cwd();run=root/'outputs/bus_height_pair_20261006_010720_56265'
    out=root/'outputs/revision_20261006_independent_tracking';out.mkdir(exist_ok=False)
    cases=json.loads((run/'summary.json').read_text())['cases'];old=json.loads((run/'protocol.json').read_text())
    checkpoint=Path('/data/models/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth')
    protocol={'model':'FasterRCNN_ResNet50_FPN_V2 COCO_V1','checkpoint_sha256':sha(checkpoint),'confidence_threshold':0.5,
              'baseline_iou_threshold':0.5,'association_iou_threshold':0.3,'max_missing_frames':0,'direction_min_consecutive_frames':3,
              'video_paths':[old['baseline_video']]+[c['generation']['artifacts']['video'] for c in cases],
              'crop':'existing evaluator crop: bottom 256 rows, x=448:896; 8 frames',
              'torchvision_version':torchvision.__version__,'thresholds_tuned_on_this_run':False,
              'source_sha256':{p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('track_audit.py')]},
              'claim':'Different detector, uncalibrated confidence vs YOLO; diagnostic support, not ground-truth accuracy. Original scores unchanged.'}
    save(out/'protocol.json',protocol)
    torch.set_num_threads(4)
    model=fasterrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None)
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True);model.eval().to('cuda')
    labels=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.meta['categories'];cache={}
    def predict(path):
        digest=sha(path)
        if digest in cache:return cache[digest]
        cap=cv2.VideoCapture(str(path));frames=[]
        while True:
            ok,frame=cap.read()
            if not ok:break
            crop=cv2.cvtColor(frame[-256:,448:896],cv2.COLOR_BGR2RGB)
            tensor=torch.from_numpy(crop).permute(2,0,1).to('cuda').float()/255
            with torch.inference_mode():pred=model([tensor])[0]
            keep=pred['scores']>=0.5
            boxes=pred['boxes'][keep].cpu().tolist();scores=pred['scores'][keep].cpu().tolist();ids=pred['labels'][keep].cpu().tolist()
            frames.append([{'label':labels[i],'score':score,'box':box} for i,score,box in zip(ids,scores,boxes)])
        cap.release();assert len(frames)==8
        save(out/f'predictions_{digest}.json',{'path':str(path),'sha256':digest,'frames':frames});cache[digest]=frames
        print('DETECTED',path,flush=True);return frames
    baseline=predict(old['baseline_video']);rows=[]
    for case in cases:
        path=case['generation']['artifacts']['video'];assert sha(path)==case['video_sha256']
        pred=predict(path);raw=[[d for d in ds if d['label']=='bus'] for ds in pred]
        residual=[[d for d in ds if not any(iou(d['box'],b['box'])>=0.5 for b in baseline[t])] for t,ds in enumerate(raw)]
        row={'mode':case['mode'],'seed':case['seed_offset'],'original_gate':case['gate'],'original_score':case['S_perc'],
             'raw_target_frames':sum(bool(ds) for ds in raw),'residual_target_frames':sum(bool(ds) for ds in residual),
             'raw_tracks':audit_tracks(raw),'residual_tracks':audit_tracks(residual),'raw':raw,'residual':residual}
        rows.append(row)
        print('RESULT',row['mode'],row['seed'],'raw/residual',row['raw_target_frames'],row['residual_target_frames'],
              'residual_track_lengths',[t['support'] for t in row['residual_tracks']['tracks']],flush=True)
    yolo=json.loads((root/'outputs/revision_20261006_height_result_audit/detections.json').read_text())
    yolo_tracks={seed:audit_tracks([f['retained_bus'] for f in item['records']]) for seed,item in yolo.items()}
    save(out/'results.json',{'protocol':protocol,'cases':rows,'original_yolo_residual_tracks':yolo_tracks,'baseline_raw_bus_frames':sum(any(d['label']=='bus' for d in fs) for fs in baseline),'output_selection_changed':False})
    del model;torch.cuda.empty_cache()
    print('COMPLETE',out,flush=True)

if __name__=='__main__':main()
