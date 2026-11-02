"""Actual geometry+loader+runner chain using labeled synthetic feedback only."""
import json,pickle,time,shutil
from pathlib import Path
from dataclasses import asdict,replace
from driveloop.native_phase import NativePhaseRegistry,NativePhaseEvaluator,NativePhaseMeasurement,digest
from driveloop.dynamic_phase import PhaseAdmission,DynamicPhaseBackend,DynamicPhaseRefiner,DynamicPhaseRunner
from driveloop.phase_input_receipt import PhaseInputReceipt
from driveloop.schema import DriveLoopConfig,Generation
from scripts.run_native_bus_control import ROOT,setup_env,make_backend
from scripts.geometry_with_source_support import check_geometry
from scripts.validate_phase_parameter import validate_phase_parameter

OUT=ROOT/'outputs/revision_20261014_dynamic_phase_audit';RANGE=ROOT/'outputs/revision_20261013_phase_range'
PRE=ROOT/'outputs/revision_20261008_candidate1583_inputs/preflight.json'
PRE_SHA='997384cdb119b1679abc84e48276da54f38e3141d297271e50e192bd499d9af2'
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,default=str)+'\n')
def main():
    OUT.mkdir(exist_ok=False);setup_env();start=time.monotonic();r=NativePhaseRegistry(PRE,PRE_SHA)
    curve=json.loads((RANGE/'results.json').read_text())['rows']
    trackpath=ROOT/'outputs/revision_20261008_signal_precheck/extended_tracks.pkl'
    source_results=json.loads((ROOT/'outputs/revision_20261008_signal_precheck/results.json').read_text())
    assert digest(trackpath)==source_results['track_cache_sha256'];track=pickle.loads(trackpath.read_bytes())[1583]
    oldreceipt=ROOT/'outputs/revision_20261014_exact_phase_input/receipt.json'
    oldsha='a40241eecda9570ba21910881bbf2bf974ac0d67e71f2c43cc2f9d6ef12eb8fc'
    source=json.loads((ROOT/'outputs/revision_20261007_native_multiscene_source/candidate1583/original/manifest.json').read_text())
    files=[ROOT/'driveloop/dynamic_phase.py',ROOT/'driveloop/phase_amplitude.py',ROOT/'scripts/validate_phase_parameter.py',ROOT/'scripts/geometry_with_source_support.py',Path(__file__).resolve(),RANGE/'results.json',trackpath]
    frozen={str(p):digest(p) for p in files}
    save(OUT/'protocol.json',{'generation_run':False,'synthetic_feedback_only':True,'targets':[-4,4],
        'synthetic_rule':'Original measurement zero; admitted nonzero action measurement equals target. Verifies plumbing only, not video response.',
        'tolerance_px':1,'max_iterations':2,'source_registry_sha256':r.fingerprint,'frozen_files':frozen,
        'preexisting_positive_receipt_sha256':oldsha,'checks':['new negative-direction phase exact admission','existing positive receipt reuse','actual loader inputs','real DriveLoopRunner state propagation','archived capture bridge']})
    def admit(phase,dest):
        dest.mkdir(parents=True,exist_ok=False)
        gm=check_geometry(source,dest/'geometry',phase,track)
        if gm is None:raise ValueError('Exact geometry rejected')
        return validate_phase_parameter(r,dest/'geometry/manifest.json',dest/'preflight',phase,files)
    admission=PhaseAdmission(r,curve,admit,OUT/'admission')
    oldphase=json.loads(oldreceipt.read_text())['phase_s'];admission.register(PhaseInputReceipt(oldreceipt,oldsha,oldphase,r))
    def factory(m,d,audit):
        assert audit
        b=make_backend(m,d,audit_only=True);b.python_executable=str(ROOT/'scripts/dd2_capture_python');return b
    results=[]
    for target in (-4,4):
        dest=OUT/('negative' if target<0 else 'positive')
        backend=DynamicPhaseBackend(admission,factory,dest/'backend',audit_only=True)
        def measure(g):
            q=g.metadata['native_phase_contract'];value=0 if q['phase_s']==0 else target
            return NativePhaseMeasurement(r.scene,r.instance,g.iteration,q['arm'],(value,)*4,{'synthetic_software_probe':True,'not_video_measurement':True},True)
        result=DynamicPhaseRunner(backend,NativePhaseEvaluator(measure,True),DynamicPhaseRefiner(admission),DriveLoopConfig(max_iterations=2,target_score=0,output_dir=dest/'history')).run(admission.request(target))
        assert result.return_status=='accepted' and len(result.attempt_history)==2
        final=result.attempt_history[-1].generation.metadata['native_phase_contract']
        assert final['phase_s']*target<0
        save(dest/'result.json',asdict(result));results.append({'target':target,'phase_sequence':[a.generation.metadata['native_phase_contract']['phase_s'] for a in result.attempt_history],'status':result.return_status,'loader_calls':len(backend.events)})
        print('DYNAMIC_CHAIN_PASS',results[-1],flush=True)
    assert len(admission.events)==1 and admission.events[0]['status']=='admitted'
    # Capture path smoke check uses an existing original video, copied provenance,
    # and read-only symlinked capture files. No old output is modified.
    archive=ROOT/'outputs/candidate1583_capture_20261009_084734_12216'
    row=next(x for x in json.loads((archive/'summary.json').read_text())['cases'] if x['seed_offset']==0 and x['arm']=='original')
    dest=OUT/'archived_capture_smoke';dest.mkdir();(dest/'capture').symlink_to(row['capture_dir'],target_is_directory=True)
    save(dest/'latent_audit.json',row['latent_audit']);q=admission.request(4).condition['native_phase']
    g=Generation(**row['generation']);g=replace(g,metadata={**g.metadata,'native_phase_contract':q})
    b=DynamicPhaseBackend(admission,lambda *a:None,OUT/'unused',audit_only=False)
    captured=b._capture(g,dest,r.case['inputs']['original']);clip=b.clip_for(captured)
    assert len(clip.frames)==8 and clip.arm=='original' and clip.seed_offset==0
    hook=json.loads((Path(row['capture_dir'])/'hook_state.json').read_text())
    assert all(f.sha256==hook['files'][f'generated_front/frame{f.index}.png'] for f in clip.frames)
    for p,h in frozen.items():assert digest(p)==h,p
    assert digest(oldreceipt)==oldsha;r.verify()
    save(OUT/'summary.json',{'passed':True,'generation_run':False,'synthetic_feedback_only':True,'cases':results,
        'new_exact_receipts':len(admission.events),'new_phase':admission.events[0]['phase_s'],
        'actual_loader_calls':6,'diffusion_calls':0,'archived_capture_bridge_passed':True,
        'elapsed_seconds':time.monotonic()-start,'limitations':'No new video generated, synthetic measurements do not demonstrate efficacy; fresh online capture remains to be exercised.'})
    print('DYNAMIC_PHASE_CHAIN_AUDIT_COMPLETE',OUT,flush=True)
if __name__=='__main__':main()
