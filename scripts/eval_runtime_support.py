"""Budget completion and fail-closed slot recovery. No inference at import time."""
import copy
import hashlib
import json
import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from driveloop.schema import Generation, Refinement

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    with temp.open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

class PresetRefiner:
    def __init__(self,requests): self.requests=copy.deepcopy(requests);self.position=0
    def refine(self,request,evaluation):
        self.position=min(self.position+1,len(self.requests)-1)
        r=self.requests[self.position]
        return Refinement(prompt=r.prompt,condition=copy.deepcopy(r.condition),notes=['predeclared_schedule; measured_diagnosis_ignored'])

def completion(result,cap=4):
    """Only the runner's explicit measurement-stop flag justifies a short arm.

    A return can be accepted from an earlier attempt while the FINAL attempt
    was unmeasurable. Check stopping and returned status separately.
    """
    attempts=result.attempt_history
    if not 1<=len(attempts)<=cap:raise ValueError('invalid attempt count')
    if [a.iteration for a in attempts]!=list(range(len(attempts))):raise ValueError('noncontiguous attempt sequence')
    flags=[bool(a.evaluation.metrics.get('perception_view_restriction_unresolved')) for a in attempts]
    if any(flags[:-1]):raise ValueError('continued after mandatory unmeasurable stop')
    if flags[-1]: reason='perception_unmeasurable'
    elif len(attempts)==cap:reason='candidate_cap_reached'
    else:raise ValueError('short arm without a permitted terminal condition')
    return {'complete':True,'termination':reason,'attempt_count':len(attempts),
            'unused_slots':cap-len(attempts),'return_status':result.return_status,
            'returned_iteration':result.returned_generation.iteration}

class InterruptedSlot(RuntimeError): pass

class SlotBackend:
    """Replay completed generation receipts; execute an incomplete slot only
    with explicit retry_incomplete. Factory receives a NEW per-try directory.
    Caller must hold an experiment lock and verify frozen code/model inputs.
    """
    def __init__(self,root,protocol_sha256,factory,retry_incomplete=False,extra_validator=None):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.protocol_sha256=protocol_sha256;self.factory=factory
        self.retry_incomplete=retry_incomplete;self.extra_validator=extra_validator
        self.events=[]

    def generate(self,request,iteration):
        key=digest({'protocol_sha256':self.protocol_sha256,'iteration':iteration,'request':asdict(request)})
        slot=self.root/f'slot_{iteration:02d}';slot.mkdir(exist_ok=True)
        receipt=slot/'receipt.json'
        if receipt.exists():
            envelope=json.loads(receipt.read_text());data=envelope['payload']
            if digest(data)!=envelope['sha256']:raise ValueError('receipt integrity failure')
            if data['key']!=key:raise ValueError('request/protocol changed for cached slot')
            for path,h in data['files'].items():
                if sha(path)!=h:raise ValueError('cached artifact changed: '+path)
            generation=Generation(**data['generation'])
            if self.extra_validator:self.extra_validator(generation,Path(data['try_directory']))
            self.events.append({'iteration':iteration,'kind':'reused_completed_slot','try_directory':data['try_directory']})
            return generation
        prior=sorted(slot.glob('try_*'))
        for p in prior:
            started=json.loads((p/'started.json').read_text())
            if started['key']!=key:raise ValueError('request/protocol changed after interrupted slot')
        if prior and not self.retry_incomplete:raise InterruptedSlot(f'{slot}: incomplete prior execution; preserve evidence and explicitly request retry of this same slot')
        dest=slot/f'try_{len(prior):03d}';dest.mkdir()
        atomic_json(dest/'started.json',{'key':key,'iteration':iteration,'request':asdict(request),'protocol_sha256':self.protocol_sha256,'time_unix':time.time(),'prior_incomplete_tries':len(prior)})
        start=time.monotonic()
        try:
            generation=self.factory(dest).generate(request,iteration)
            if generation.iteration!=iteration:raise ValueError('backend returned wrong iteration')
            if self.extra_validator:self.extra_validator(generation,dest)
            files={str(p.resolve()):sha(p) for p in sorted(dest.rglob('*')) if p.is_file()}
            for value in generation.artifacts.values():
                if isinstance(value,str) and Path(value).is_file():files[str(Path(value).resolve())]=sha(value)
            data={'key':key,'generation':asdict(generation),'try_directory':str(dest),'files':files,'wall_seconds':time.monotonic()-start}
            atomic_json(receipt,{'payload':data,'sha256':digest(data)})
            self.events.append({'iteration':iteration,'kind':'new_completed_slot','try_directory':str(dest),'prior_incomplete_tries':len(prior),'wall_seconds':data['wall_seconds']})
            return generation
        except Exception as e:
            atomic_json(dest/'failed.json',{'type':type(e).__name__,'message':str(e),'wall_seconds':time.monotonic()-start})
            raise
