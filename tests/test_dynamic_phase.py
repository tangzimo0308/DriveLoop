from dataclasses import replace
import copy,json
from pathlib import Path
import pytest
from tests.test_native_phase import registry
from tests.test_phase_input_receipt import packet
from driveloop.dynamic_phase import PhaseAdmission,DynamicPhaseBackend,DynamicPhaseRefiner,DynamicPhaseRunner,action_id
from driveloop.phase_input_receipt import PhaseInputReceipt
from driveloop.native_phase import NativePhaseEvaluator,NativePhaseMeasurement,digest
from driveloop.schema import Generation,DriveLoopConfig

CURVE=[{'phase_s':i/8,'source_mean_dx_px':-50*i/8,'geometry_passed':True} for i in range(-4,5)]
def manager(r,path,admit):return PhaseAdmission(r,CURVE,admit,path)

def fake_receipt(r,phase,path):
    path.mkdir(parents=True);m=path/'geometry.json';ds=path/'dataset';ds.mkdir()
    m.write_text(json.dumps({'geometry':{'passed':True},'selected_delay_seconds':phase,'instance_token':r.instance,'original_manifest':{'scene_token':r.scene},'native_dataset':r.case['inputs']['original']['dataset'],'edited_dataset':str(ds)}))
    audit=copy.deepcopy(r.case['runtime_audits']['original']);audit['box_downsampler_input']['sha256']=phase.hex()
    d={'schema':'exact_phase_input_v1','passed':True,'generation_run':False,'phase_s':phase,'registry_sha256':r.fingerprint,'scene_token':r.scene,'instance_token':r.instance,'candidate_index':r.candidate,'geometry_manifest':str(m),'input_manifest':{'dataset':str(ds)},'fixed_hashes':{str(m):digest(m)},'geometry_passed':True,'raster_passed':True,'runtime_passed':True,'first_frames_unchanged':True,'runtime_audit':audit}
    p=path/'receipt.json';p.write_text(json.dumps(d));return PhaseInputReceipt(p,digest(p),phase,r)

def backend_factory(a,events):
    def factory(m,d,audit_only):
        matches=[p for p in a.receipts.values() if p.data['input_manifest']['dataset']==m['dataset']]
        receipt=matches[0] if matches else None
        expected=receipt.data['runtime_audit'] if receipt else a.source.case['runtime_audits']['original']
        events.append('factory')
        class Backend:
            def generate(self,request,i):
                assert not request.condition and not request.metadata
                return Generation(i,request.prompt,metadata={'dd2_source_sample_binding':{'ready':True,'dataset_dir':m['dataset'],'selector':{'scene_token':a.source.scene,'instance_token':a.source.instance}},'dd2_audit_only':audit_only,'dd2_prompt':None,'dd2_override_json':{'available':False},'dd2_runtime_input_audit':copy.deepcopy(expected)})
        return Backend()
    return factory

@pytest.mark.parametrize('target,phase',[(4,-.08),(-4,.08)])
def test_real_runner_admits_exact_proposal_before_next_backend(registry,tmp_path,target,phase):
    r=registry;events=[]
    def admit(p,d):events.append('admission');return fake_receipt(r,p,d)
    a=manager(r,tmp_path/'admit',admit);b=DynamicPhaseBackend(a,backend_factory(a,events),tmp_path/'out',True)
    def measure(g):
        q=g.metadata['native_phase_contract'];return NativePhaseMeasurement(r.scene,r.instance,g.iteration,q['arm'],(-50*q['phase_s'],)*4,{'unit_test':True},True)
    result=DynamicPhaseRunner(b,NativePhaseEvaluator(measure,True),DynamicPhaseRefiner(a),DriveLoopConfig(max_iterations=3,target_score=0,output_dir=tmp_path/'history')).run(a.request(target))
    assert events==['factory','admission','factory']
    assert result.attempt_history[1].generation.metadata['native_phase_contract']['phase_s']==pytest.approx(phase)
    assert result.return_status=='accepted'

def test_rejected_action_never_calls_backend_and_not_retried(registry,tmp_path):
    events=[]
    def reject(p,d):events.append('reject');raise ValueError('geometry rejected')
    a=manager(registry,tmp_path/'a',reject);b=DynamicPhaseBackend(a,backend_factory(a,events),tmp_path/'b',True)
    for i in range(2):
        with pytest.raises(ValueError):b.generate(a.request(4,phase=-.08),i)
    assert events==['reject'] and len(a.events)==1

def test_wrong_exact_receipt_rejected_before_backend(registry,tmp_path):
    events=[]
    a=manager(registry,tmp_path/'a',lambda p,d:fake_receipt(registry,p+.001,d));b=DynamicPhaseBackend(a,backend_factory(a,events),tmp_path/'b',True)
    with pytest.raises(ValueError):b.generate(a.request(4,phase=-.08),0)
    assert events==[]

def test_missing_measurement_stops_without_admission(registry,tmp_path):
    r=registry;events=[]
    def reject(p,d):raise AssertionError('must not admit')
    a=manager(r,tmp_path/'a',reject)
    def measure(g):return NativePhaseMeasurement(r.scene,r.instance,g.iteration,'original',(None,)*4,{'unit_test':True},True)
    result=DynamicPhaseRunner(DynamicPhaseBackend(a,backend_factory(a,events),tmp_path/'b',True),NativePhaseEvaluator(measure,True),DynamicPhaseRefiner(a),DriveLoopConfig(target_score=0,output_dir=tmp_path/'h')).run(a.request(4))
    assert events==['factory'] and result.return_status=='perception_unmeasurable'

def test_cached_receipt_tamper_rejected(registry,tmp_path):
    a=manager(registry,tmp_path/'a',lambda p,d:fake_receipt(registry,p,d));receipt=a.resolve(-.08)
    Path(receipt.data['geometry_manifest']).write_text('{}')
    with pytest.raises(ValueError):a.resolve(-.08)

@pytest.mark.parametrize('phase',[.6,-.6,float('nan')])
def test_out_of_range_or_nan_rejected(registry,tmp_path,phase):
    a=manager(registry,tmp_path/'a',lambda p,d:None)
    with pytest.raises(ValueError):a.contract(a.request(4,phase=phase))

def test_action_mismatch(registry,tmp_path):
    a=manager(registry,tmp_path/'a',lambda p,d:None);req=a.request(4,phase=-.08)
    req=replace(req,condition={'native_phase':{**req.condition['native_phase'],'arm':'original'}})
    with pytest.raises(ValueError):a.contract(req)
