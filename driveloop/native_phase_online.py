"""Opt-in development pilot helpers. No changes to the default DriveLoop runner."""
from dataclasses import asdict, replace
from pathlib import Path
import json
from driveloop.native_phase import NativePhaseRefiner, NativePhaseEvaluator, digest
from driveloop.native_phase_png import PNGFrame, PNGClip, FixedBusDetector
from driveloop.schema import Refinement

METHODS = ('resample_original', 'direct_target', 'fixed_schedule', 'feedback')
SCHEDULE = ('original', 'delayed', 'advanced')

class PilotRefiner(NativePhaseRefiner):
    def __init__(self, registry, method, target_arm):
        super().__init__(registry)
        if method not in METHODS or target_arm not in ('delayed', 'advanced'):
            raise ValueError('Unknown pilot policy')
        self.method, self.target_arm = method, target_arm

    def initial_arm(self):
        return self.target_arm if self.method == 'direct_target' else 'original'

    def refine(self, request, evaluation):
        if self.method == 'feedback':
            return super().refine(request, evaluation)
        q = self.registry.contract(request)
        f = evaluation.metrics['native_phase_feedback']
        if f['contract'] != q:
            raise ValueError('Measurement contract mismatch')
        # These baselines do not use the measured values to choose their action.
        if self.method == 'fixed_schedule':
            arm = SCHEDULE[min(f['iteration']+1, len(SCHEDULE)-1)]
        else:
            arm = self.target_arm if self.method == 'direct_target' else 'original'
        return Refinement(request.prompt, {'native_phase': {**q, 'arm': arm}}, ['preset_policy:'+self.method])

class ReleasingDetector:
    """Do not hold a detector on the GPU while the generator subprocess runs."""
    def __init__(self, weight):
        from driveloop.native_phase_png import DETECTOR_SHA
        self.weight, self.model = weight, None
        self.provenance = {'detector':'FasterRCNN_ResNet50_FPN_V2_COCO_V1','weight_sha256':DETECTOR_SHA,
                           'confidence':.5,'unique_bus_iou':.3,'device':'cuda'}
    def __call__(self, image):
        if self.model is None:
            self.model = FixedBusDetector(self.weight)
        return self.model(image)
    def release_gpu(self):
        if self.model is not None:
            del self.model
            self.model = None
            import torch
            torch.cuda.empty_cache()

class OnlineEvaluator(NativePhaseEvaluator):
    def __init__(self, callback, detector):
        super().__init__(callback)
        self.detector = detector
    def evaluate(self, generation):
        try:
            return super().evaluate(generation)
        finally:
            self.detector.release_gpu()


def make_clip(row, registry, plans, evidence_path):
    """Authenticate fresh generation, capture, seed and source before measuring."""
    from scripts.run_native_matched_capture import validate_hooks
    g = row['generation']; meta = g['metadata']; arm = row['arm']; seed = row['seed_offset']
    if arm not in SCHEDULE or g['iteration'] != seed:
        raise ValueError('Invalid generated action or seed')
    binding = meta['dd2_source_sample_binding']
    if not binding['ready'] or binding['selector']['scene_token'] != registry.scene or binding['selector']['instance_token'] != registry.instance:
        raise ValueError('Actual source binding differs')
    if meta['dd2_audit_only'] or meta['dd2_prompt'] is not None or meta['dd2_override_json']['available']:
        raise ValueError('Not a native generated video')
    for key, expected in registry.case['runtime_audits'][arm].items():
        if key in ('prompt_embed','img_cond','grounding_downsampler_input','box_downsampler_input'):
            if meta['dd2_runtime_input_audit'][key]['sha256'] != expected['sha256']:
                raise ValueError('Input tensor differs')
    latent = row['latent_audit']
    if latent['scale'] != 1 or latent['image_latent_before'] != latent['image_latent_after']:
        raise ValueError('Image conditioning changed')
    if digest(g['artifacts']['video']) != row['video_sha256']:
        raise ValueError('Output video changed')
    capture = Path(row['capture_dir']); hook = validate_hooks(capture)
    frames = tuple(PNGFrame(i,str(capture/f'generated_front/frame{i}.png'),hook['files'][f'generated_front/frame{i}.png'],tuple(plans[arm][i])) for i in range(8))
    return PNGClip(registry.scene,registry.instance,registry.fingerprint,seed,arm,latent['initial_noise']['sha256'],frames,str(evidence_path),digest(evidence_path))


def method_summary(result, method, target_arm):
    """Primary output is continuous error and completeness, not test acceptance."""
    attempts = []
    for a in result.attempt_history:
        f = a.evaluation.metrics['native_phase_feedback']
        attempts.append({'iteration':a.iteration,'arm':f['contract']['arm'], 'observed_dx_px':f['observed_dx_px'],
                         'absolute_error_px':abs(f['error_px']) if f['error_px'] is not None else None,
                         'measured':f['status']=='measured','video':a.generation.artifacts['video']})
    measured = [a for a in attempts if a['measured']]
    best = min(measured,key=lambda a:a['absolute_error_px']) if measured else None
    return {'method':method,'target_arm':target_arm,'attempts':attempts,'best_measured':best,
            'complete_three_candidate_budget':len(attempts)==3,
            'all_attempts_measurable':all(a['measured'] for a in attempts),
            'logical_reference_renders':3,'actual_candidate_renders':len(attempts),
            'logical_total_renders':3+len(attempts),'logical_render_cap':6,
            'paper_success_rate_computed':False}
