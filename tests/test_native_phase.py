import copy,json,os
from dataclasses import replace
from pathlib import Path
import pytest
from driveloop.native_phase import *
from driveloop.schema import Generation,DriveLoopConfig
@pytest.fixture
def registry(tmp_path,monkeypatch):
 for k,v in {'DRIVELOOP_EGO_INJECTION':'0','DRIVELOOP_DD2_IMAGE_CONDITION_SCALE':'1','DRIVELOOP_DD2_NUM_INF_STEPS':'50','DRIVELOOP_DD2_SEED_BANK':'0'}.items():monkeypatch.setenv(k,v)
 sig=lambda h:{'sha256':h};inputs={};audits={};arms={}
 for arm,x in [('original',0),('delayed',-12),('advanced',14)]:
  path=tmp_path/arm;path.write_text(arm)
  inputs[arm]={'scene_token':'scene','instance_token':'bus','candidate_index':1,'dataset':str(path)}
  audits[arm]={k:sig(arm if k=='box_downsampler_input' else k) for k in SIGNATURES}
  arms[arm]={'geometry':{'passed':True,'frames':[{'projected_center_shift_x_px_at448':x} for _ in range(8)]}}
 manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'selected':[{'arms':arms}]}));protocol=tmp_path/'protocol.json';protocol.write_text('{}');weight=tmp_path/'weight';weight.write_bytes(b'weight')
 pre={'passed':True,'generation_run':False,'cases':[{'scene_token':'scene','candidate_index':1,'inputs':inputs,'runtime_audits':audits}], 'fixed_hashes':{str(p):digest(p) for p in (manifest,protocol,weight,*[tmp_path/a for a in ARMS])},'weight':str(weight),'weight_sha256':digest(weight)}
 path=tmp_path/'preflight.json';path.write_text(json.dumps(pre));return NativePhaseRegistry(path,digest(path))
def generation(r,q,i=0):return Generation(i,r.request(0,1).prompt,metadata={'native_phase_contract':q})
def evaluate(r,q,values,synthetic=False):
 return NativePhaseEvaluator(lambda g:NativePhaseMeasurement(r.scene,r.instance,g.iteration,q['arm'],tuple(values),{'test_fixture':True},synthetic),allow_synthetic=synthetic).evaluate(generation(r,q))
@pytest.mark.parametrize('target,arm',[(14,'advanced'),(-12,'delayed')])
def test_signed_error_routes_to_approved_action(registry,target,arm):
 r=registry;request=r.request(target,1);e=evaluate(r,request.condition['native_phase'],[0]*4);ref=NativePhaseRefiner(r).refine(request,e);assert ref.condition['native_phase']['arm']==arm
@pytest.mark.parametrize('bad',[float('nan'),float('inf'),True,'1'])
def test_nonfinite_target_rejected(registry,bad):
 with pytest.raises(ValueError):registry.request(bad,1)
def test_nonpositive_tolerance_rejected(registry):
 with pytest.raises(ValueError):registry.request(0,0)
def test_no_action_outside_registry(registry):
 r=registry;req=r.request(100,1);q={**req.condition['native_phase'],'arm':'advanced'};req=replace(req,condition={'native_phase':q});ref=NativePhaseRefiner(r).refine(req,evaluate(r,q,[14]*4));assert ref.condition['native_phase']['arm']=='advanced';assert 'hold_no_better_approved_action' in ref.notes
@pytest.mark.parametrize('change',[{'arm':'custom'},{'scene_token':'other'},{'instance_token':'other'},{'registry_sha256':'wrong'}])
def test_wrong_route_contract_rejected(registry,change):
 req=registry.request(14,1);req=replace(req,condition={'native_phase':{**req.condition['native_phase'],**change}})
 with pytest.raises(ValueError):registry.contract(req)
def test_hash_tamper_rejected(registry):
 Path(registry.case['inputs']['advanced']['dataset']).write_text('tampered')
 with pytest.raises(ValueError):registry.verify()
def test_feedback_from_other_action_rejected(registry):
 r=registry;req=r.request(14,1);e=evaluate(r,{**req.condition['native_phase'],'arm':'delayed'},[0]*4)
 with pytest.raises(ValueError):NativePhaseRefiner(r).refine(req,e)
def test_unmeasured_not_zero(registry):
 r=registry;q=r.request(14,1).condition['native_phase'];e=evaluate(r,q,[0,None,0,0]);f=e.metrics['native_phase_feedback'];assert f['observed_dx_px'] is None and f['error_px'] is None and e.metrics['native_phase_unmeasurable']
def test_synthetic_disabled_by_default(registry):
 r=registry;q=r.request(14,1).condition['native_phase'];ev=NativePhaseEvaluator(lambda g:NativePhaseMeasurement(r.scene,r.instance,0,'original',(0,0,0,0),{'test':True},True))
 with pytest.raises(ValueError):ev.evaluate(generation(r,q))
@pytest.mark.parametrize('bad',[(0,0,0),(0,float('nan'),0,0)])
def test_invalid_measurements_rejected(registry,bad):
 with pytest.raises(ValueError):evaluate(registry,registry.request(1,1).condition['native_phase'],bad)
def test_wrong_seed_measurement_rejected(registry):
 r=registry;q=r.request(1,1).condition['native_phase'];ev=NativePhaseEvaluator(lambda g:NativePhaseMeasurement(r.scene,r.instance,99,'original',(0,0,0,0),{'test':True}))
 with pytest.raises(ValueError):ev.evaluate(generation(r,q))
def factory(r,calls,tamper=False):
 def build(m,d,audit):
  arm=next(a for a in ARMS if r.case['inputs'][a]['dataset']==m['dataset']);calls.append(arm)
  class Backend:
   def generate(self,req,i):
    assert req.condition=={} and req.metadata=={}
    hashes=copy.deepcopy(r.case['runtime_audits'][arm])
    if tamper:hashes['img_cond']['sha256']='bad'
    return Generation(i,req.prompt,metadata={'dd2_runtime_input_audit':hashes,'dd2_prompt':None,'dd2_override_json':{'available':False},'dd2_audit_only':audit,'dd2_source_sample_binding':{'ready':True,'selector':{'scene_token':r.scene,'instance_token':r.instance}}})
  return Backend()
 return build
@pytest.mark.parametrize('missing',[False,True])
def test_real_runner_feedback_routes_or_stops(registry,tmp_path,missing):
 r=registry;calls=[];b=NativePhaseBackend(r,factory(r,calls),tmp_path/'outputs',True)
 def measure(g):
  arm=g.metadata['native_phase_contract']['arm'];values=(None,0,0,0) if missing else (r.means[arm],)*4
  return NativePhaseMeasurement(r.scene,r.instance,g.iteration,arm,values,{'unit_test':True},True)
 e=NativePhaseEvaluator(measure,True);ref=NativePhaseRefiner(r);config=DriveLoopConfig(max_iterations=3,target_score=0,output_dir=tmp_path/'history')
 result=NativePhaseRunner(b,e,ref,config).run(r.request(14,1));assert calls==(['original'] if missing else ['original','advanced']);assert len(result.history)==len(calls)
 assert result.return_status==('perception_unmeasurable' if missing else 'accepted')
def test_actual_input_hash_mismatch_rejected(registry,tmp_path):
 b=NativePhaseBackend(registry,factory(registry,[],True),tmp_path/'out',True)
 with pytest.raises(ValueError,match='Actual input differs'):b.generate(registry.request(14,1),0)
def test_conflicting_controls_rejected(registry):
 req=registry.request(14,1);req=replace(req,condition={**req.condition,'structural_escalation':{'size_scale':2}})
 with pytest.raises(ValueError):registry.contract(req)
