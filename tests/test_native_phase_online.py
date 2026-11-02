from dataclasses import replace
import pytest
from tests.test_native_phase import registry, factory
from driveloop.native_phase import NativePhaseBackend, NativePhaseRunner, NativePhaseEvaluator, NativePhaseMeasurement
from driveloop.native_phase_online import METHODS, PilotRefiner, OnlineEvaluator, method_summary
from driveloop.schema import DriveLoopConfig

@pytest.mark.parametrize('method,expected', [
    ('resample_original',['original']*3),('direct_target',['advanced']*3),
    ('fixed_schedule',['original','delayed','advanced']),('feedback',['original','advanced','advanced'])])
def test_real_runner_policy_and_budget(registry,tmp_path,method,expected):
    r=registry;calls=[];base=factory(r,calls)
    def build(m,d,audit):
        original=base(m,d,audit)
        class Backend:
            def generate(self,request,i):
                return replace(original.generate(request,i),artifacts={'video':str(d/'video.mp4')})
        return Backend()
    backend=NativePhaseBackend(r,build,tmp_path/'candidates',True)
    def measure(g):
        arm=g.metadata['native_phase_contract']['arm']
        return NativePhaseMeasurement(r.scene,r.instance,g.iteration,arm,(r.means[arm],)*4,{'fixture':True},True)
    refiner=PilotRefiner(r,method,'advanced');request=r.request(14,1)
    request=replace(request,condition={'native_phase':{**request.condition['native_phase'],'arm':refiner.initial_arm()}})
    result=NativePhaseRunner(backend,NativePhaseEvaluator(measure,True),refiner,DriveLoopConfig(max_iterations=3,target_score=0,stop_on_acceptance=False,output_dir=tmp_path/'history')).run(request)
    assert calls==expected
    s=method_summary(result,method,'advanced')
    assert s['logical_total_renders']==6 and s['actual_candidate_renders']==3
    assert s['complete_three_candidate_budget'] and s['all_attempts_measurable']
    assert s['best_measured']['absolute_error_px']==(14 if method=='resample_original' else 0)

@pytest.mark.parametrize('method',METHODS)
def test_unmeasurable_stops_and_budget_is_not_falsely_full(registry,tmp_path,method):
    r=registry;calls=[];base=factory(r,calls)
    def build(m,d,audit):
        b=base(m,d,audit)
        class Backend:
            def generate(self,request,i):return replace(b.generate(request,i),artifacts={'video':str(d/'v.mp4')})
        return Backend()
    def measure(g):
        arm=g.metadata['native_phase_contract']['arm']
        vals=(None,)*4 if g.iteration==1 else (0,)*4
        return NativePhaseMeasurement(r.scene,r.instance,g.iteration,arm,vals,{'fixture':True},True)
    ref=PilotRefiner(r,method,'advanced');req=r.request(14,1);req=replace(req,condition={'native_phase':{**req.condition['native_phase'],'arm':ref.initial_arm()}})
    result=NativePhaseRunner(NativePhaseBackend(r,build,tmp_path/'out',True),NativePhaseEvaluator(measure,True),ref,DriveLoopConfig(max_iterations=3,target_score=0,stop_on_acceptance=False,output_dir=tmp_path/'history')).run(req)
    s=method_summary(result,method,'advanced')
    assert len(calls)==2 and s['logical_total_renders']==5
    assert not s['complete_three_candidate_budget'] and not s['all_attempts_measurable']
    assert s['attempts'][1]['absolute_error_px'] is None

def test_detector_released_on_measurement_error():
    class Detector:
        released=False
        def release_gpu(self):self.released=True
    def fail(g):raise ValueError('bad capture')
    d=Detector();e=OnlineEvaluator(fail,d)
    from driveloop.schema import Generation
    with pytest.raises(ValueError):e.evaluate(Generation(0,'',metadata={'native_phase_contract':{}}))
    assert d.released
