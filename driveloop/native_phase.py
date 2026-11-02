"""Opt-in native phase routing. This diagnostic does not implement the paper's utility J.

No default runner behavior changes. Callers supply an independently defined target,
measurement callback and tolerance; the approved finite action set is never expanded.
"""
from __future__ import annotations
import copy,hashlib,json,math,os
from dataclasses import dataclass,replace
from pathlib import Path
from statistics import mean
from driveloop.schema import DriveLoopRequest,Evaluation,Diagnosis,Refinement
from driveloop.runner import DriveLoopRunner
ARMS=('original','delayed','advanced')
SIGNATURES=('prompt_embed','img_cond','grounding_downsampler_input','box_downsampler_input')
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def number(value):
 if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):raise ValueError('Expected a finite numeric value')
 return float(value)
class NativePhaseRegistry:
 def __init__(self,path,expected_sha256):
  self.path=Path(path);self.fingerprint=expected_sha256
  if digest(self.path)!=expected_sha256:raise ValueError('Unapproved preflight digest')
  self.pre=json.loads(self.path.read_text());self.verify()
  if not self.pre.get('passed') or self.pre.get('generation_run') or len(self.pre.get('cases',[]))!=1:raise ValueError('One approved ungenerated preflight required')
  self.case=self.pre['cases'][0]
  manifest_path=self.path.parent/'manifest.json';protocol_path=self.path.parent/'protocol.json'
  for p in (manifest_path,protocol_path):
   if self.pre['fixed_hashes'].get(str(p))!=digest(p):raise ValueError('Unpinned manifest or protocol')
  selected=json.loads(manifest_path.read_text())['selected']
  if len(selected)!=1:raise ValueError('One selected scene required')
  self.plan=selected[0];self.protocol=json.loads(protocol_path.read_text())
  self.scene=self.case['scene_token'];self.instance=self.case['inputs']['original']['instance_token'];self.candidate=self.case['candidate_index']
  self.means={'original':0.0}
  if set(self.case['inputs'])!=set(ARMS):raise ValueError('Unexpected action inventory')
  for arm in ARMS:
   m=self.case['inputs'][arm]
   if m['scene_token']!=self.scene or m['instance_token']!=self.instance or m['candidate_index']!=self.candidate:raise ValueError('Mixed scene, target or candidate')
   a=self.case['runtime_audits'][arm];original=self.case['runtime_audits']['original']
   if any(a[k]['sha256']!=original[k]['sha256'] for k in SIGNATURES[:-1]):raise ValueError('Non-pose input differs')
   if arm!='original':
    if a[SIGNATURES[-1]]['sha256']==original[SIGNATURES[-1]]['sha256']:raise ValueError('Edited action must change boxes')
    geometry=self.plan['arms'][arm]['geometry']
    if not geometry['passed'] or len(geometry['frames'])!=8:raise ValueError('Geometry not approved')
    self.means[arm]=mean(number(f['projected_center_shift_x_px_at448']) for f in geometry['frames'][4:8])
  if self.means['delayed']*self.means['advanced']>=0:raise ValueError('Opposite signed phase effects required')
 def verify(self):
  if digest(self.path)!=self.fingerprint:raise ValueError('Preflight changed')
  for path,h in self.pre['fixed_hashes'].items():
   if digest(path)!=h:raise ValueError('Frozen input changed: '+path)
  if digest(self.pre['weight'])!=self.pre['weight_sha256']:raise ValueError('Model weight changed')
 def request(self,target_dx_px,tolerance_px):
  target=number(target_dx_px);tol=number(tolerance_px)
  if tol<=0:raise ValueError('Positive explicit tolerance required')
  return DriveLoopRequest(prompt='Native-source bus retention diagnostic; use unchanged DD2 source conditions.',condition={'native_phase':{'registry_sha256':self.fingerprint,'scene_token':self.scene,'instance_token':self.instance,'candidate_index':self.candidate,'arm':'original','target_dx_px':target,'tolerance_px':tol}})
 def contract(self,request):
  if set(request.condition)-{'native_phase','dd2_condition'}:raise ValueError('Unsupported concurrent control')
  q=request.condition.get('native_phase');base=self.request(0,1).condition['native_phase']
  if not isinstance(q,dict) or set(q)!=set(base):raise ValueError('Invalid native phase contract')
  if any(q[k]!=base[k] for k in ('registry_sha256','scene_token','instance_token','candidate_index')):raise ValueError('Unapproved identity or registry')
  if request.prompt!=self.request(0,1).prompt:raise ValueError('Native phase diagnostic uses fixed cached-source prompt')
  if q['arm'] not in ARMS:raise ValueError('Unapproved phase action')
  number(q['target_dx_px'])
  if number(q['tolerance_px'])<=0:raise ValueError('Invalid tolerance')
  # Source hints are not applied by this restricted backend; refuse attempted rebinding.
  for k in ('scene_token','instance_token','source_candidate_id','source_rebinding'):
   if k in request.metadata:raise ValueError('External source rebinding is not supported')
  return copy.deepcopy(q)
class NativePhaseBackend:
 def __init__(self,registry,factory,output_dir,audit_only=False):
  self.registry=registry;self.factory=factory;self.output_dir=Path(output_dir);self.audit_only=audit_only
 def generate(self,request,iteration):
  q=self.registry.contract(request);self.registry.verify()
  for k,v in {'DRIVELOOP_EGO_INJECTION':'0','DRIVELOOP_DD2_IMAGE_CONDITION_SCALE':'1','DRIVELOOP_DD2_NUM_INF_STEPS':'50','DRIVELOOP_DD2_SEED_BANK':'0'}.items():
   if os.environ.get(k)!=v:raise ValueError('Unexpected runtime setting: '+k)
  dest=self.output_dir/f'iteration_{iteration:02d}_{q["arm"]}';dest.mkdir(parents=True,exist_ok=False)
  manifest=copy.deepcopy(self.registry.case['inputs'][q['arm']]);backend=self.factory(manifest,dest,self.audit_only)
  # This explicit mode uses only the audited native inputs. Generic prompt/actor plans
  # made by the outer runner are not passed through as additional DD2 overrides.
  fixed=replace(request,condition={},metadata={})
  g=backend.generate(fixed,iteration);meta=g.metadata;a=meta['dd2_runtime_input_audit'];expected=self.registry.case['runtime_audits'][q['arm']]
  binding=meta.get('dd2_source_sample_binding',{})
  if g.iteration!=iteration or not binding.get('ready') or binding.get('selector',{}).get('scene_token')!=q['scene_token'] or binding.get('selector',{}).get('instance_token')!=q['instance_token']:raise ValueError('Actual generation binding differs')
  if meta.get('dd2_prompt') is not None or meta['dd2_override_json']['available']:raise ValueError('Unexpected text or structural override')
  for k in SIGNATURES:
   if a[k]['sha256']!=expected[k]['sha256']:raise ValueError('Actual input differs: '+k)
  if bool(meta.get('dd2_audit_only'))!=self.audit_only:raise ValueError('Backend execution mode differs')
  self.registry.verify()
  return replace(g,metadata={**meta,'native_phase_contract':q,'native_phase_route':{'dataset':manifest['dataset'],'arm':q['arm'],'registry_sha256':self.registry.fingerprint,'runtime_inputs_match':True,'derived_generic_dd2_plan_applied':False}})
@dataclass(frozen=True)
class NativePhaseMeasurement:
 scene_token:str
 instance_token:str
 seed_offset:int
 arm:str
 # Four same-seed edited-minus-original center deltas for frames 4,5,6,7;
 # None means absent/ambiguous on either side, never zero imputation.
 late_deltas:tuple
 provenance:dict
 synthetic:bool=False
class NativePhaseEvaluator:
 def __init__(self,measure,allow_synthetic=False):self.measure=measure;self.allow_synthetic=allow_synthetic
 def evaluate(self,generation):
  q=generation.metadata['native_phase_contract'];m=self.measure(generation)
  if not isinstance(m,NativePhaseMeasurement):raise ValueError('Typed phase measurement required')
  if m.synthetic and (not self.allow_synthetic or not generation.metadata.get('dd2_audit_only')):raise ValueError('Synthetic feedback requires explicitly enabled audit-only execution')
  if (m.scene_token,m.instance_token,m.seed_offset,m.arm)!=(q['scene_token'],q['instance_token'],generation.iteration,q['arm']):raise ValueError('Measurement identity/seed/action mismatch')
  if len(m.late_deltas)!=4 or not m.provenance:raise ValueError('Four-frame provenance required')
  values=[number(v) if v is not None else None for v in m.late_deltas];complete=all(v is not None for v in values)
  observed=mean(values) if complete else None;error=q['target_dx_px']-observed if complete else None
  passed=complete and abs(error)<=q['tolerance_px']
  feedback={'contract':q,'iteration':generation.iteration,'status':'measured' if complete else 'unmeasured','late_indices':[4,5,6,7],'late_deltas':values,'observed_dx_px':observed,'error_px':error,'provenance':copy.deepcopy(m.provenance),'synthetic':m.synthetic}
  return Evaluation(score=1/(1+abs(error)) if complete else 0,metrics={'native_phase_feedback':feedback,'native_phase_unmeasurable':not complete},diagnosis=Diagnosis(passed=passed,reasons=[] if passed else ['native_phase_error' if complete else 'native_phase_unmeasured']))
class NativePhaseRefiner:
 def __init__(self,registry):self.registry=registry
 def refine(self,request,evaluation):
  q=self.registry.contract(request);f=evaluation.metrics.get('native_phase_feedback')
  if not isinstance(f,dict) or f.get('contract')!=q:raise ValueError('Feedback/goal/action contract mismatch')
  arm=q['arm'];reason='hold_unmeasured'
  if f.get('status')=='measured':
   error=number(f['error_px']);observed=number(f['observed_dx_px'])
   if abs((q['target_dx_px']-observed)-error)>1e-9:raise ValueError('Inconsistent signed residual')
   if abs(error)<=q['tolerance_px']:reason='hold_within_tolerance'
   else:
    desired=self.registry.means[arm]+error
    # A nearest approved action is a hypothesis based on source geometry,
    # never a claim about the next generated video or a continuous controller.
    ordered=[arm]+[a for a in ARMS if a!=arm]
    proposed=min(ordered,key=lambda a:abs(self.registry.means[a]-desired))
    if proposed==arm:reason='hold_no_better_approved_action'
    else:arm=proposed;reason='choose_nearest_approved_expected_shift'
  elif f.get('status')!='unmeasured':raise ValueError('Unknown measurement status')
  condition={k:copy.deepcopy(v) for k,v in request.condition.items() if k!='dd2_condition'};condition['native_phase']={**q,'arm':arm}
  return Refinement(prompt=request.prompt,condition=condition,notes=[reason,'approved_native_phase:'+arm])
class NativePhaseRunner(DriveLoopRunner):
 def __init__(self,backend,evaluator,refiner,config):
  if not isinstance(backend,NativePhaseBackend) or not isinstance(evaluator,NativePhaseEvaluator) or not isinstance(refiner,NativePhaseRefiner):raise ValueError('Explicit native phase components required')
  if backend.registry.fingerprint!=refiner.registry.fingerprint:raise ValueError('Different action registries')
  if config.use_task_utility or config.use_postprocess or config.acceptance_policy!='score_and_diagnosis' or config.target_score!=0:raise ValueError('Native phase diagnostic requires explicit diagnostic-only scoring config')
  super().__init__(backend=backend,evaluator=evaluator,refiner=refiner,config=config)
 @staticmethod
 def _perception_unmeasurable(evaluation):
  return bool(evaluation.metrics.get('native_phase_unmeasurable')) or DriveLoopRunner._perception_unmeasurable(evaluation)
