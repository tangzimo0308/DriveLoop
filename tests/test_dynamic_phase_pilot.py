import pytest
from tests.test_native_phase import registry
from tests.test_dynamic_phase import fake_receipt,backend_factory,manager
from driveloop.dynamic_phase import DynamicPhaseBackend,DynamicPhaseRunner
from driveloop.dynamic_phase_pilot import AmplitudePilotRefiner
from driveloop.native_phase import NativePhaseEvaluator,NativePhaseMeasurement
from driveloop.schema import DriveLoopConfig

@pytest.mark.parametrize('method,expected',[
 ('direct_target',[-.25,-.25,-.25]),('fixed_amplitude_search',[-.25,-.125,-.375]),('residual_feedback',[-.25,-.3,-.3])])
def test_equal_start_and_three_rounds(registry,tmp_path,method,expected):
 r=registry;a=manager(r,tmp_path/'admit',lambda p,d:fake_receipt(r,p,d));events=[]
 def measure(g):
  q=g.metadata['native_phase_contract']
  return NativePhaseMeasurement(r.scene,r.instance,g.iteration,q['arm'],(-40*q['phase_s'],)*4,{'unit_test':True},True)
 runner=DynamicPhaseRunner(DynamicPhaseBackend(a,backend_factory(a,events),tmp_path/'out',True),NativePhaseEvaluator(measure,True),AmplitudePilotRefiner(a,method,-.25),DriveLoopConfig(max_iterations=3,target_score=0,stop_on_acceptance=False,output_dir=tmp_path/'hist'))
 result=runner.run(a.request(12.5,phase=-.25))
 assert [t.generation.metadata['native_phase_contract']['phase_s'] for t in result.attempt_history]==pytest.approx(expected)
 assert events.count('factory')==3
