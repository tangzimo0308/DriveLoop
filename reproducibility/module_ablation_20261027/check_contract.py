"""CPU-only validator checks using copied frozen metadata, not new video results."""
import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import run_module_ablation as m

f=m.unseal(m.ref.OUT/'freeze.json');group=f['groups'][0]
case,bank=group['case_id'],group['bank']
receipt=m.unseal(m.ref.OUT/'groups'/case/str(bank)/'feedback/slots/slot_00/receipt.json')
dest=Path(receipt['try_directory']);request=m.read(dest/'started.json')['request']
gen=m.Generation(**copy.deepcopy(receipt['generation']))
source=gen.metadata['dd2_runtime_input_audit']
pre=m.read(m.ref.SOURCE/'source_preflight.json')
manifest=m.read(next(c['manifest'] for c in pre['cases'] if c['case_id']==case))
noise=m.read(dest/'latent_audit.json')['initial_noise']
plan=m.read(m.PRE/'request_plan.json');observed=m.read(m.PRE/'observed_inputs.json')
known={r['request_sha256']:observed[k]['input_audit'] for k,r in plan['unique_requests'].items()
       if r['case_id']==case and r['backend_mode']=='frozen_reference'}
assert m.digest(request) in known
checked=[]
with tempfile.TemporaryDirectory(prefix='contract_only_',dir=m.OUT) as tmp:
    path=Path(tmp)
    m.atomic_json(path/'started.json',{'request':request})
    validate=m.validator(manifest,bank,'without_size',path/'group',known,source,{0:noise})
    # Capture integrity is separately enforced by the unchanged frozen validator;
    # inject a recorded signature here to test the new pairing/comparison layer.
    with patch.object(m.ref,'validate_capture',return_value=noise):
        validate(gen,path)
        checked.append('recorded_slot0_satisfies_no_size_input_contract')
        for name in ('noise_mismatch','source_change','tensor_drift','size_not_removed'):
            modified=copy.deepcopy(gen)
            req=copy.deepcopy(request)
            fake_noise=copy.deepcopy(noise)
            if name=='noise_mismatch':fake_noise['sha256']='0'*64
            elif name=='source_change':modified.metadata['dd2_source_sample_binding']['selector']['instance_token']='wrong'
            elif name=='tensor_drift':modified.metadata['dd2_runtime_input_audit']['box_downsampler_input']['sha256']='0'*64
            else:req['condition']['structural_escalation']={'size_scale':1.5}
            m.atomic_json(path/'started.json',{'request':req})
            try:
                with patch.object(m.ref,'validate_capture',return_value=fake_noise):validate(modified,path)
            except AssertionError:checked.append(name+'_rejected')
            else:raise AssertionError('Validator accepted '+name)
    assert not list((m.OUT/'groups').glob('*/*/*/slots/slot_*/receipt.json'))
result={'checks':checked,'cpu_only':True,'new_videos':0,
        'capture_signature_mocked_for_contract_test':True,
        'claim':'Validator contract tested; no actual diffusion/noise pairing run yet.'}
m.atomic_json(m.OUT/'contract_checks.json',result)
print(json.dumps(result,indent=2))
