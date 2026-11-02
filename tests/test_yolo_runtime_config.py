"""Threshold semantics and provenance at the inference/runner boundary."""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import pytest
from driveloop.perception_video import UltralyticsYOLODetector, PerceptionVideoEvaluator
from driveloop.schema import DriveLoopConfig, DriveLoopRequest, Evaluation, Generation
from driveloop.runner import DriveLoopRunner

class FakeModel:
    def __init__(self):
        self.calls=[];self.predictor=SimpleNamespace(args=SimpleNamespace())
    def predict(self,source,verbose,conf):
        self.calls.append(conf)
        self.predictor.args=SimpleNamespace(conf=conf,iou=.7,imgsz=640,device='cpu',half=False)
        class XY:
            def __getitem__(self,i):return self
            def tolist(self):return [0.,0.,20.,20.]
        boxes=[SimpleNamespace(conf=[c],cls=[0],xyxy=XY()) for c in [.22,.27,.35] if c>=conf]
        return [SimpleNamespace(names={0:'bus'},boxes=boxes)]

@pytest.fixture
def factory(monkeypatch,tmp_path):
    weights=tmp_path/'fake.pt';weights.write_bytes(b'weights identity')
    monkeypatch.setattr(UltralyticsYOLODetector,'_load_model',lambda self:FakeModel())
    return lambda **kwargs:UltralyticsYOLODetector(weights,**kwargs)

def test_default_retains_upstream_025_when_post_filter_is_02(factory):
    d=factory(confidence_threshold=.2);boxes=d.detect('unused.jpg',0)
    assert [b.confidence for b in boxes]==[.27,.35]
    assert d.model.calls==[.25]

def test_explicit_02_reaches_model_and_post_filter_is_independent(factory):
    d=factory(confidence_threshold=.2,inference_confidence_threshold=.2)
    assert [b.confidence for b in d.detect('unused.jpg',0)]==[.22,.27,.35]
    d=factory(confidence_threshold=.3,inference_confidence_threshold=.2)
    assert [b.confidence for b in d.detect('unused.jpg',0)]==[.35]

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-.1,1.1])
@pytest.mark.parametrize('parameter',['confidence_threshold','inference_confidence_threshold'])
def test_invalid_threshold_fails_before_loading(monkeypatch,bad,parameter):
    monkeypatch.setattr(UltralyticsYOLODetector,'_load_model',lambda self:pytest.fail('must not load invalid config'))
    with pytest.raises(ValueError):UltralyticsYOLODetector('unused',**{parameter:bad})

def test_runtime_is_detached_and_survives_release_and_reload(factory):
    d=factory(confidence_threshold=.2);d.detect('unused.jpg',0);audit=d.get_runtime_audit()
    settings=audit['variants'][0]['settings']
    assert settings['effective_predictor_args']['conf']==.25
    assert settings['post_filter_confidence_threshold']==.2
    assert len(settings['model_identity']['weights_sha256'])==64
    d.model.predictor.args.conf=.9
    assert audit['variants'][0]['settings']['effective_predictor_args']['conf']==.25
    d.release_gpu();assert d.get_runtime_audit()==audit
    d.detect('unused.jpg',1);assert d.get_runtime_audit()['successful_predict_calls']==2
    assert d.model.calls==[.25]
    d.begin_runtime_audit();assert d.get_runtime_audit()['status']=='not_invoked'
    assert d.get_runtime_audit()['variants']==[]
    json.dumps(audit,allow_nan=False)

def test_runtime_mismatch_fails_closed(factory):
    d=factory();predict=d.model.predict
    def wrong(**kwargs):
        result=predict(**kwargs);d.model.predictor.args.conf=.1;return result
    d.model.predict=wrong
    with pytest.raises(RuntimeError,match='differs'):d.detect('unused.jpg',0)

def test_unavailable_effective_args_are_not_invented(factory):
    d=factory();d.model.predict=lambda **kwargs:[];d.model.predictor=None
    d.detect('unused.jpg',0)
    assert d.get_runtime_audit()['variants'][0]['settings']['effective_predictor_args'] is None

def test_standalone_metadata_report_clears_previous_inference(factory):
    d=factory();d.detect('unused.jpg',0)
    g=Generation(0,'bus',metadata={'perception_detections':{'frames':[], 'frame_count':0}})
    report=PerceptionVideoEvaluator(detector=d).build_report(g)
    assert report['detector_runtime']['status']=='not_invoked'
    assert report['detector_runtime']['successful_predict_calls']==0

class Backend:
    def generate(self,request,iteration):return Generation(iteration,request.prompt)
class Evaluator:
    confidence_threshold=.2
    tracker_iou_threshold=.3
    def __init__(self,detector):self.detector=detector
    def evaluate(self,generation):
        if generation.iteration==0:self.detector.detect('unused.jpg',0)
        return Evaluation(0.)

def test_runner_archives_each_attempt_before_release_and_no_stale_calls(factory,tmp_path):
    d=factory(confidence_threshold=.2)
    r=DriveLoopRunner(backend=Backend(),evaluator=Evaluator(d),config=DriveLoopConfig(max_iterations=2,target_score=.99,output_dir=tmp_path/'run')).run(DriveLoopRequest('a bus'))
    assert len(r.attempt_history)==2
    first=r.attempt_history[0].generation.metadata['perception_detector_runtime']
    second=r.attempt_history[1].generation.metadata['perception_detector_runtime']
    assert first['successful_predict_calls']==1
    assert first['evaluator_post_filter_confidence_threshold']==.2
    assert first['variants'][0]['settings']['effective_predictor_args']['conf']==.25
    assert second['status']=='not_invoked' and second['variants']==[]
    assert d.model is None
    saved=[json.loads(s) for s in (tmp_path/'run/history.jsonl').read_text().splitlines()]
    assert saved[0]['generation']['metadata']['perception_detector_runtime']==first
    assert saved[1]['generation']['metadata']['perception_detector_runtime']==second
