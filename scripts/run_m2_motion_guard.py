"""Matched M2 control with lateral geometry guard v1; controls are declared before any generated evaluations.

Baseline is an uninjected source-scene render used only as evaluator support.
All three experimental arms use four seeds, 50 steps, and a shared window.
No paper-level claim follows from a single-window pilot.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

from driveloop.backends.drivedreamer2 import DriveDreamer2Backend
from driveloop.refiner import RuleBasedRefiner
from driveloop.runner import DriveLoopRunner
from driveloop.schema import Diagnosis, DriveLoopConfig, DriveLoopRequest, Evaluation, Refinement

PROMPT = 'rainy urban street, a truck cuts in from the left toward the ego vehicle'


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def schedules(prompt=PROMPT):
    initial = DriveLoopRequest(prompt=prompt, condition={
        'driveloop_original_prompt': prompt, 'generation_escalation': {'num_inf_steps': 50}})
    # A declared recipe-building stimulus, NOT a measured result. It is never
    # passed into the runner or written as an experimental evaluation.
    stimulus = Evaluation(score=0.0, diagnosis=Diagnosis(passed=False,
        reasons=['target_object_not_detected', 'low_detection_coverage']))
    fixed = [initial]
    recipe_builder = RuleBasedRefiner()
    for _ in range(3):
        refinement = recipe_builder.refine(fixed[-1], stimulus)
        fixed.append(replace(initial, prompt=refinement.prompt, condition=refinement.condition))
    direct = [copy.deepcopy(fixed[2]) for _ in range(4)]
    return {'fixed': fixed, 'direct': direct, 'full': [initial]}


class PresetRefiner:
    """Ignores both the measured evaluation and incoming request."""
    def __init__(self, requests):
        self.requests = copy.deepcopy(requests)
        self.position = 0

    def refine(self, request, evaluation):
        self.position = min(self.position + 1, len(self.requests) - 1)
        planned = self.requests[self.position]
        return Refinement(prompt=planned.prompt, condition=copy.deepcopy(planned.condition),
                          notes=['predeclared_control; measured_feedback_not_used'])


class TimedBackend(DriveDreamer2Backend):
    def generate(self, request, iteration):
        start = time.monotonic()
        generation = super().generate(request, iteration)
        return replace(generation, metadata={**generation.metadata,
            'generation_wall_seconds': time.monotonic() - start})


def code_hashes(root):
    paths = sorted((root / 'driveloop').rglob('*.py'))
    paths += [root / 'scripts/run_m2_motion_guard.py',
              root / 'dreamer-train/projects/DriveDreamer2/drivedreamer2/drivedreamer2_tester.py']
    return {str(p.relative_to(root)): sha256(p) for p in paths if p.is_file()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--identity', type=Path, required=True)
    parser.add_argument('--window-manifest', type=Path, required=True)
    parser.add_argument('--stage', choices=['prepare', 'baseline', 'direct', 'fixed', 'full'], required=True)
    parser.add_argument('--seed-bank', type=int, default=0)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    out = args.output_dir.resolve()
    dataset, identity = args.dataset.resolve(), args.identity.resolve()
    out.mkdir(parents=True, exist_ok=True)
    weight = Path('/data/models/dd2/pretrained_models/drivedreamer2_img_cond/pytorch_gligen_weights.bin')
    # Do not inherit optional ablation controls or stale prompt/override input.
    for name in list(os.environ):
        if name.startswith('DRIVELOOP_'):
            del os.environ[name]
    os.environ.update({
        'DRIVELOOP_DD2_WEIGHT_PATH': str(weight),
        'DRIVELOOP_DD2_NUM_INF_STEPS': '50',
        'DRIVELOOP_DD2_SEED_BANK': str(args.seed_bank),
        'DRIVELOOP_EGO_INJECTION': '1',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
    })
    window = json.loads(args.window_manifest.read_text())
    assert window['ready'] is True
    plan = schedules(window['prompt'])
    manifest_path = out / 'protocol.json'
    if args.stage == 'prepare':
        if manifest_path.exists():
            raise FileExistsError('Protocol already exists; use a new output directory')
        subset_report = json.loads((dataset / 'subset_report.json').read_text())
        assert subset_report['source_candidate_index'] == window['candidate_index']
        assert subset_report['subset_binding']['ready'] is True
        assert subset_report['subset_record_count'] == 144
        manifest = {
            'protocol_version': 'm2_lateral_geometry_v1',
            'source_code_sha256': code_hashes(root),
            'guard': {'version': 'lateral_geometry_v1', 'minimum_lateral_meters': 0.5,
                      'claim': 'necessary geometric compatibility, not semantic success'},
            'status': 'declared_before_generation', 'prompt': window['prompt'], 'window': window,
            'seed_bank': args.seed_bank, 'seed_offsets': [args.seed_bank * 100 + i for i in range(4)],
            'num_inference_steps': 50, 'dataset': str(dataset), 'identity': str(identity),
            'dataset_labels_sha256': sha256(dataset / 'labels/data.pkl'),
            'identity_sha256': sha256(identity), 'dd2_weight_sha256': sha256(weight),
            'detector_weights_sha256': sha256(root / 'yolov8x.pt'),
            'script_sha256': sha256(Path(__file__)),
            'evaluator': 'ManeuverViewRestrictedSuperclassEvaluator (v10b)',
            'detector_confidence': 0.20, 'acceptance_policy': 'joint_perception_direction',
            'perception_threshold': 0.3, 'stop_on_acceptance': False,
            'utility_weights': {'perception': 0.5, 'control': 0.3, 'intent': 0.2},
            'fixed_schedule_definition': 'Recipe generated once before rendering with the declared target-not-detected/low-coverage stimulus; all actual feedback ignored.',
            'direct_definition': 'Fixed schedule rung 2 used on all four seeds; no actual feedback.',
            'full_definition': 'Existing RuleBasedRefiner receives measured feedback.',
            'planned_requests': {k: [asdict(r) for r in v] for k, v in plan.items()},
            'historical_A10_identity_verified': False,
            'python': platform.python_version(),
            'code_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        }
        write_json(manifest_path, manifest)
        print('PROTOCOL_SAVED', manifest_path, flush=True)
        return
    manifest = json.loads(manifest_path.read_text())
    assert manifest['source_code_sha256'] == code_hashes(root), 'Code changed after declaration'
    assert manifest['dd2_weight_sha256'] == sha256(weight), 'DD2 weights changed'
    assert manifest['detector_weights_sha256'] == sha256(root / 'yolov8x.pt'), 'Detector weights changed'
    assert manifest['window'] == window
    assert manifest['seed_bank'] == args.seed_bank
    assert manifest['script_sha256'] == sha256(Path(__file__)), 'Harness changed after declaration'
    assert manifest['dataset_labels_sha256'] == sha256(dataset / 'labels/data.pkl')
    assert manifest['identity_sha256'] == sha256(identity)
    arm_dir = out / args.stage
    if arm_dir.exists():
        raise FileExistsError(f'{arm_dir} already exists; refusing to overwrite')
    arm_dir.mkdir()
    backend = TimedBackend(project_root=root, baseline_dataset_dir=dataset,
        artifact_dir=arm_dir / 'artifacts', timeout_seconds=3600,
        source_candidate_id=window['source_candidate_id'], scene_token=window['scene_token'],
        instance_token=window['instance_token'], source_identity_summary_path=identity)
    start = time.monotonic()
    if args.stage == 'baseline':
        generation = backend.generate(DriveLoopRequest(prompt='Uninjected source-scene support baseline',
            condition={'generation_escalation': {'num_inf_steps': 50}}), iteration=0)
        assert not generation.metadata.get('dd2_override_json', {}).get('available', False)
        write_json(arm_dir / 'generation.json', asdict(generation))
        write_json(arm_dir / 'timing.json', {'wall_seconds': time.monotonic() - start})
    else:
        from driveloop.perception_video import UltralyticsYOLODetector
        from driveloop.perception_v10 import ManeuverViewRestrictedSuperclassEvaluator
        baseline = json.loads((out / 'baseline/generation.json').read_text())
        baseline_video = baseline['artifacts']['video']
        assert Path(baseline_video).is_file()
        evaluator = ManeuverViewRestrictedSuperclassEvaluator(
            detector=UltralyticsYOLODetector(str(root / 'yolov8x.pt'), confidence_threshold=0.20),
            confidence_threshold=0.20, baseline_video=baseline_video)
        runner = DriveLoopRunner(backend=backend, evaluator=evaluator,
            refiner=(RuleBasedRefiner() if args.stage == 'full' else PresetRefiner(plan[args.stage])),
            config=DriveLoopConfig(max_iterations=4, target_score=0.99, output_dir=arm_dir / 'history',
                use_task_utility=True, acceptance_policy='joint_perception_direction',
                perception_acceptance_threshold=0.3, stop_on_acceptance=False))
        result = runner.run(plan[args.stage][0])
        write_json(arm_dir / 'result.json', asdict(result))
        write_json(arm_dir / 'timing.json', {'wall_seconds': time.monotonic() - start,
            'baseline_sha256': sha256(baseline_video), 'attempt_count': len(result.attempt_history)})
        assert len(result.attempt_history) == 4, 'Incomplete budget; do not report as a completed arm'
    print('STAGE_COMPLETE', args.stage, 'seconds', round(time.monotonic() - start, 2), flush=True)


if __name__ == '__main__':
    main()
