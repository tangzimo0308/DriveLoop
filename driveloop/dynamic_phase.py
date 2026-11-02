"""Opt-in dynamic amplitude loop with admission before backend invocation."""
import copy, hashlib, json, os, time
from dataclasses import asdict, replace
from pathlib import Path
from driveloop.native_phase import number, digest, NativePhaseEvaluator, NativePhaseRunner, SIGNATURES
from driveloop.phase_amplitude import propose_phase
from driveloop.phase_input_receipt import PhaseInputReceipt
from driveloop.schema import DriveLoopRequest, Refinement
from driveloop.runner import DriveLoopRunner


def action_id(phase):
    phase=number(phase)
    return 'original' if phase == 0 else 'phase_'+hashlib.sha256(phase.hex().encode()).hexdigest()[:16]

def save(path,obj):Path(path).write_text(json.dumps(obj,indent=2,default=str)+'\n')

class PhaseAdmission:
    """Admission callback must construct and check the exact proposed parameter."""
    def __init__(self, source, curve, admit_exact, output_dir):
        self.source,self.curve,self.admit_exact=source,copy.deepcopy(curve),admit_exact
        self.output_dir=Path(output_dir);self.output_dir.mkdir(parents=True,exist_ok=False)
        self.receipts={};self.events=[];self.failed=set()
    def request(self,target,tolerance=1,phase=0):
        q=self.source.request(target,tolerance).condition['native_phase']
        phase=number(phase);q.update(arm=action_id(phase),phase_s=phase)
        return DriveLoopRequest(self.source.request(0,1).prompt,condition={'native_phase':q})
    def contract(self,request):
        q=request.condition.get('native_phase');base=self.request(0).condition['native_phase']
        if not isinstance(q,dict) or set(q)!=set(base) or set(request.condition)-{'native_phase','dd2_condition'}:
            raise ValueError('Invalid dynamic phase contract')
        if request.prompt!=self.request(0).prompt:raise ValueError('Fixed cached native prompt required')
        for k in ('scene_token','instance_token','candidate_index','registry_sha256'):
            if q[k]!=base[k]:raise ValueError('Dynamic source mismatch')
        phase=number(q['phase_s']);number(q['target_dx_px'])
        if number(q['tolerance_px'])<=0 or q['arm']!=action_id(phase):raise ValueError('Phase/action or tolerance mismatch')
        if not self.curve[0]['phase_s']<=phase<=self.curve[-1]['phase_s']:raise ValueError('Phase outside declared range')
        if any(k in request.metadata for k in ('scene_token','instance_token','source_candidate_id','source_rebinding')):
            raise ValueError('Source rebinding forbidden')
        return copy.deepcopy(q)
    def register(self,receipt):
        receipt.verify()
        if receipt.source_registry.fingerprint!=self.source.fingerprint:raise ValueError('Other source receipt')
        self.receipts[action_id(receipt.phase_s)]=receipt
    def resolve(self,phase):
        self.source.verify();phase=number(phase);key=action_id(phase)
        if not self.curve[0]['phase_s']<=phase<=self.curve[-1]['phase_s']:raise ValueError('Out of range')
        if phase==0:return None
        if key in self.failed:raise ValueError('Previously rejected phase; automatic retry forbidden')
        if key not in self.receipts:
            dest=self.output_dir/key
            save(self.output_dir/(key+'_requested.json'),{'phase_s':phase,'action_id':key,'registry_sha256':self.source.fingerprint})
            start=time.monotonic()
            try:
                receipt=self.admit_exact(phase,dest)
                if not isinstance(receipt,PhaseInputReceipt) or receipt.phase_s.hex()!=phase.hex():raise ValueError('Wrong exact receipt returned')
                self.register(receipt)
                self.events.append({'phase_s':phase,'action_id':key,'status':'admitted','receipt':str(receipt.path),'receipt_sha256':receipt.expected_sha256,'seconds':time.monotonic()-start})
            except Exception as exc:
                self.failed.add(key);self.events.append({'phase_s':phase,'action_id':key,'status':'rejected','error':str(exc),'seconds':time.monotonic()-start})
                save(self.output_dir/'admission_ledger.json',self.events);raise
            save(self.output_dir/'admission_ledger.json',self.events)
        result=self.receipts[key];result.verify();return result

class DynamicPhaseRefiner:
    def __init__(self,admission):self.admission=admission
    def refine(self,request,evaluation):
        q=self.admission.contract(request);f=evaluation.metrics['native_phase_feedback']
        if f['contract']!=q:raise ValueError('Feedback contract mismatch')
        p=propose_phase(self.admission.curve,q['phase_s'],q['target_dx_px'],f['late_deltas'],max_step_s=.125,deadband_px=q['tolerance_px'])
        if f['status']=='measured':
            if p['residual_px'] is None or abs(p['residual_px']-number(f['error_px']))>1e-9:raise ValueError('Residual inconsistent')
        elif f['status']!='unmeasured' or p['status']!='hold_unmeasured':raise ValueError('Inconsistent missing measurement')
        phase=p['proposed_phase_s']
        return Refinement(request.prompt,{'native_phase':{**q,'phase_s':phase,'arm':action_id(phase)}},[json.dumps(p,sort_keys=True)])

class DynamicPhaseBackend:
    def __init__(self,admission,factory,output_dir,audit_only=False):
        self.admission,self.factory,self.output_dir,self.audit_only=admission,factory,Path(output_dir),audit_only
        self.clips={};self.capture_rows={};self.events=[]
    def generate(self,request,iteration):
        q=self.admission.contract(request)
        for k,v in {'DRIVELOOP_EGO_INJECTION':'0','DRIVELOOP_DD2_IMAGE_CONDITION_SCALE':'1','DRIVELOOP_DD2_NUM_INF_STEPS':'50','DRIVELOOP_DD2_SEED_BANK':'0'}.items():
            if os.environ.get(k)!=v:raise ValueError('Runtime setting differs: '+k)
        # Admission finishes BEFORE the generation backend factory is called.
        receipt=self.admission.resolve(q['phase_s'])
        manifest=copy.deepcopy(receipt.data['input_manifest'] if receipt else self.admission.source.case['inputs']['original'])
        expected=receipt.data['runtime_audit'] if receipt else self.admission.source.case['runtime_audits']['original']
        dest=self.output_dir/f'iteration_{iteration:02d}_{q["arm"]}';dest.mkdir(parents=True,exist_ok=False)
        backend=self.factory(manifest,dest,self.audit_only)
        env={'DRIVELOOP_CAPTURE_DIR':str(dest/'capture'),'DRIVELOOP_DD2_IMAGE_CONDITION_AUDIT_PATH':str(dest/'latent_audit.json')}
        old={k:os.environ.get(k) for k in env};os.environ.update(env);start=time.monotonic()
        try:g=backend.generate(replace(request,condition={},metadata={}),iteration)
        finally:
            for k,v in old.items():
                if v is None:os.environ.pop(k,None)
                else:os.environ[k]=v
        if g.iteration!=iteration:raise ValueError('Generation seed/iteration mismatch')
        m=g.metadata;b=m['dd2_source_sample_binding'];s=b['selector']
        if not b['ready'] or (s['scene_token'],s['instance_token'])!=(q['scene_token'],q['instance_token']):raise ValueError('Actual identity mismatch')
        if Path(b['dataset_dir']).resolve()!=Path(manifest['dataset']).resolve():raise ValueError('Actual dataset mismatch')
        if m['dd2_prompt'] is not None or m['dd2_override_json']['available'] or bool(m['dd2_audit_only'])!=self.audit_only:raise ValueError('Unexpected override or mode')
        for k in SIGNATURES:
            if m['dd2_runtime_input_audit'][k]['sha256']!=expected[k]['sha256']:raise ValueError('Runtime tensor mismatch: '+k)
        if receipt:receipt.verify_runtime(g,audit_only=self.audit_only)
        else:self.admission.source.verify()
        metadata={**m,'native_phase_contract':q,'dynamic_phase_receipt':str(receipt.path) if receipt else None,
                  'dynamic_phase_receipt_sha256':receipt.expected_sha256 if receipt else None,'dynamic_phase_input_verified':True}
        g=replace(g,metadata=metadata)
        if self.audit_only:
            if 'video' in g.artifacts:raise ValueError('Audit-only run generated video')
        else:
            g=self._capture(g,dest,manifest)
        self.events.append({'iteration':iteration,'phase_s':q['phase_s'],'arm':q['arm'],'audit_only':self.audit_only,'seconds':time.monotonic()-start,'receipt':metadata['dynamic_phase_receipt']})
        save(self.output_dir/'execution_ledger.json',self.events)
        return g
    def _capture(self,g,dest,manifest):
        from scripts.run_native_matched_capture import validate_hooks
        from driveloop.native_phase_png import PNGFrame,PNGClip
        import pickle,numpy as np
        from dreamer_datasets import boxes3d_utils
        hook=validate_hooks(dest/'capture');latent=json.loads((dest/'latent_audit.json').read_text())
        if latent['scale']!=1 or latent['image_latent_before']!=latent['image_latent_after']:raise ValueError('Image conditioning modified')
        q=g.metadata['native_phase_contract'];records=pickle.loads((Path(manifest['dataset'])/'labels/data.pkl').read_bytes());frames=[]
        for i in range(8):
            r=records[24+i*3];j=r['instance_tokens'].index(q['instance_token'])
            corners=boxes3d_utils.boxes3d_to_corners3d(r['boxes3d'][j:j+1],rot_axis=1)
            if corners[...,2].min()<=.5:raise ValueError('Invalid projection')
            pts=boxes3d_utils.corners3d_to_corners2d(corners,cam_intrinsic=r['calib']['cam_intrinsic'])[0].astype(np.float64)*np.array([448/1600,256/900])
            box=tuple(np.r_[pts.min(0),pts.max(0)].tolist())
            frames.append(PNGFrame(i,str(dest/f'capture/generated_front/frame{i}.png'),hook['files'][f'generated_front/frame{i}.png'],box))
        row={'generation':asdict(g),'latent_audit':latent,'capture_dir':str(dest/'capture'),'video_sha256':digest(g.artifacts['video'])}
        path=dest/'capture_receipt.json';save(path,row);key=str(path)
        self.capture_rows[key]=row
        self.clips[key]=PNGClip(q['scene_token'],q['instance_token'],q['registry_sha256'],g.iteration,q['arm'],latent['initial_noise']['sha256'],tuple(frames),key,digest(path))
        return replace(g,artifacts={**g.artifacts,'native_phase_receipt':key})
    def clip_for(self,g):
        key=g.artifacts['native_phase_receipt'];row=self.capture_rows[key]
        if row['generation']['metadata']['native_phase_contract']!=g.metadata['native_phase_contract'] or row['generation']['iteration']!=g.iteration or row['generation']['artifacts']['video']!=g.artifacts['video']:
            raise ValueError('Capture receipt/generation mismatch')
        return self.clips[key]

class DynamicPhaseRunner(DriveLoopRunner):
    def __init__(self,backend,evaluator,refiner,config):
        if not isinstance(backend,DynamicPhaseBackend) or not isinstance(refiner,DynamicPhaseRefiner) or not isinstance(evaluator,NativePhaseEvaluator):raise ValueError('Explicit dynamic components required')
        if backend.admission is not refiner.admission:raise ValueError('Different admission managers')
        if config.use_task_utility or config.use_postprocess or config.target_score!=0 or config.acceptance_policy!='score_and_diagnosis':raise ValueError('Diagnostic-only config required')
        super().__init__(backend=backend,evaluator=evaluator,refiner=refiner,config=config)
    _perception_unmeasurable=staticmethod(NativePhaseRunner._perception_unmeasurable)
