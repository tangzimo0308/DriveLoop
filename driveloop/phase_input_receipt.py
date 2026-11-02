"""Exact-parameter input receipts; never expand admission to nearby phases.

A receipt certifies one source-bound, hash-frozen loader input. It does not prove
generated-video efficacy or authorize an untested continuous parameter interval.
"""
import json
from pathlib import Path
from driveloop.native_phase import digest, number, SIGNATURES

class PhaseInputReceipt:
    def __init__(self, path, expected_sha256, phase_s, source_registry):
        self.path = Path(path)
        self.expected_sha256 = expected_sha256
        self.phase_s = number(phase_s)
        self.source_registry = source_registry
        self.verify()

    def verify(self):
        if digest(self.path) != self.expected_sha256:
            raise ValueError('Input receipt changed')
        self.source_registry.verify()
        data = json.loads(self.path.read_text())
        if data.get('schema') != 'exact_phase_input_v1' or not data.get('passed') or data.get('generation_run'):
            raise ValueError('Completed input-only preflight required')
        if number(data['phase_s']).hex() != self.phase_s.hex():
            raise ValueError('Receipt does not cover the requested exact phase')
        r = self.source_registry
        if (data['registry_sha256'], data['scene_token'], data['instance_token'], data['candidate_index']) != (r.fingerprint,r.scene,r.instance,r.candidate):
            raise ValueError('Receipt bound to different source')
        for path, h in data['fixed_hashes'].items():
            if digest(path) != h:
                raise ValueError('Frozen action artifact changed: '+path)
        manifest_path = data['geometry_manifest']
        if manifest_path not in data['fixed_hashes']:
            raise ValueError('Geometry manifest must be pinned')
        m = json.loads(Path(manifest_path).read_text())
        if not m['geometry']['passed'] or number(m['selected_delay_seconds']).hex() != self.phase_s.hex():
            raise ValueError('Exact geometry phase differs')
        if m['instance_token'] != r.instance or m['original_manifest']['scene_token'] != r.scene:
            raise ValueError('Geometry identity differs')
        if Path(m['native_dataset']).resolve() != Path(r.case['inputs']['original']['dataset']).resolve():
            raise ValueError('Geometry native dataset differs')
        if data['input_manifest']['dataset'] != m['edited_dataset']:
            raise ValueError('Geometry/loader dataset mismatch')
        for name in ('geometry_passed','raster_passed','runtime_passed','first_frames_unchanged'):
            if data.get(name) is not True:
                raise ValueError('Incomplete action validation: '+name)
        baseline = r.case['runtime_audits']['original']
        for key in SIGNATURES[:-1]:
            if data['runtime_audit'][key]['sha256'] != baseline[key]['sha256']:
                raise ValueError('Non-pose runtime input changed')
        if data['runtime_audit'][SIGNATURES[-1]]['sha256'] == baseline[SIGNATURES[-1]]['sha256']:
            raise ValueError('No runtime box change observed')
        self.data = data
        return data

    def verify_runtime(self, generation, *, audit_only=True):
        self.verify()
        meta = generation.metadata
        if bool(meta.get('dd2_audit_only')) != audit_only:
            raise ValueError('Unexpected backend execution mode')
        if audit_only and 'video' in generation.artifacts:
            raise ValueError('Input-only run produced a video')
        binding = meta['dd2_source_sample_binding']
        selector = binding.get('selector', {})
        if not binding.get('ready') or (selector.get('scene_token'),selector.get('instance_token')) != (self.data['scene_token'],self.data['instance_token']):
            raise ValueError('Runtime identity differs')
        if Path(binding['dataset_dir']).resolve() != Path(self.data['input_manifest']['dataset']).resolve():
            raise ValueError('Runtime dataset differs')
        if meta.get('dd2_prompt') is not None or meta['dd2_override_json']['available']:
            raise ValueError('Unexpected prompt or structural override')
        for key in SIGNATURES:
            if meta['dd2_runtime_input_audit'][key]['sha256'] != self.data['runtime_audit'][key]['sha256']:
                raise ValueError('Runtime input differs: '+key)
        return True
