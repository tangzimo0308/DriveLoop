"""Lossless PNG measurements for native phase diagnostics.

Clip providers must bind these records to verified generation/capture manifests.
This module measures projected image displacement, not physical motion or identity.
It never generates videos and never selects a fallback detection.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import copy

from driveloop.native_phase import NativePhaseMeasurement, digest, number

DETECTOR_SHA = 'dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf'

@dataclass(frozen=True)
class PNGFrame:
    index: int
    path: str
    sha256: str
    projected_box: tuple

@dataclass(frozen=True)
class PNGClip:
    scene_token: str
    instance_token: str
    registry_sha256: str
    seed_offset: int
    arm: str
    initial_noise_sha256: str
    frames: tuple
    evidence_path: str
    evidence_sha256: str


def box(value):
    if len(value) != 4:
        raise ValueError('Four box coordinates required')
    x1, y1, x2, y2 = map(number, value)
    if x2 <= x1 or y2 <= y1:
        raise ValueError('Invalid bounding box')
    return x1, y1, x2, y2


def iou(a, b):
    a, b = box(a), box(b)
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union


def associate(raw, planned_box):
    """Fixed prior rules: bus, confidence >= .5, IoU >= .3, exactly one."""
    box(planned_box)
    matches = []
    for item in raw:
        coords = box(item['box'])
        score = number(item['score'])
        if not 0 <= score <= 1:
            raise ValueError('Invalid detector confidence')
        if item['label'] == 'bus' and score >= .5 and iou(coords, planned_box) >= .3:
            matches.append(item)
    return {'association_count': len(matches), 'target': copy.deepcopy(matches[0]) if len(matches) == 1 else None}


class FixedBusDetector:
    """Load the already installed, hash-pinned detector; never download weights."""
    def __init__(self, weight_path, device='cuda'):
        if digest(weight_path) != DETECTOR_SHA:
            raise ValueError('Detector checkpoint differs')
        import torch
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
        self.torch, self.device = torch, device
        self.model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
        self.model.load_state_dict(torch.load(weight_path, map_location='cpu', weights_only=True))
        self.model.eval().to(device)
        self.labels = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.meta['categories']
        self.provenance = {'detector': 'FasterRCNN_ResNet50_FPN_V2_COCO_V1', 'weight_sha256': DETECTOR_SHA, 'confidence': .5, 'unique_bus_iou': .3, 'device': device}

    def __call__(self, image):
        import numpy as np
        torch = self.torch
        x = torch.from_numpy(np.array(image).copy()).permute(2, 0, 1).to(self.device).float()/255
        with torch.inference_mode():
            pred = self.model([x])[0]
        ids = (pred['scores'] >= .5).nonzero().flatten()
        return [{'label': self.labels[int(pred['labels'][j])], 'score': float(pred['scores'][j]), 'box': pred['boxes'][j].cpu().tolist()} for j in ids]


class PNGPhaseMeasurement:
    """Typed callback for NativePhaseEvaluator, with explicit same-seed references.

The providers authenticate clip provenance and its link to the generation record.
Hash checking is repeated even when detections are cached. The cache saves detector
work, not the logical generation cost of each original reference.
"""
    def __init__(self, clip_for_generation, reference_for_generation, detector):
        self.clip_for_generation = clip_for_generation
        self.reference_for_generation = reference_for_generation
        self.detector = detector
        self.cache = {}
        self.detector_calls = 0

    def detect_clip(self, clip):
        from PIL import Image
        if digest(clip.evidence_path) != clip.evidence_sha256:
            raise ValueError('Capture evidence changed')
        if tuple(f.index for f in clip.frames) != tuple(range(8)):
            raise ValueError('Exactly all eight ordered frames required')
        if len({str(Path(f.path).resolve()) for f in clip.frames}) != 8:
            raise ValueError('Duplicate frame path')
        records = []
        for frame in clip.frames:
            if Path(frame.path).suffix.lower() != '.png' or digest(frame.path) != frame.sha256:
                raise ValueError('PNG capture changed or missing')
            box(frame.projected_box)
            key = (frame.sha256, tuple(frame.projected_box))
            with Image.open(frame.path) as im:
                if im.format != 'PNG' or im.mode != 'RGB' or im.size != (448, 256):
                    raise ValueError('Expected unresized RGB front capture 448x256')
                if key not in self.cache:
                    raw = self.detector(im)
                    self.cache[key] = {'raw': raw, **associate(raw, frame.projected_box)}
                    self.detector_calls += 1
            records.append({'frame': frame.index, 'png_sha256': frame.sha256, **copy.deepcopy(self.cache[key])})
        return records

    def __call__(self, generation):
        q = generation.metadata['native_phase_contract']
        current = self.clip_for_generation(generation)
        reference = self.reference_for_generation(generation)
        if not isinstance(current, PNGClip) or not isinstance(reference, PNGClip):
            raise ValueError('Verified PNGClip records required')
        for clip in (current, reference):
            if (clip.scene_token, clip.instance_token, clip.registry_sha256, clip.seed_offset) != (q['scene_token'], q['instance_token'], q['registry_sha256'], generation.iteration):
                raise ValueError('Clip identity, registry or seed mismatch')
        if current.arm != q['arm'] or reference.arm != 'original':
            raise ValueError('Wrong current action or original reference')
        if not current.initial_noise_sha256 or current.initial_noise_sha256 != reference.initial_noise_sha256:
            raise ValueError('Reference initial noise mismatch')
        a, b = self.detect_clip(reference), self.detect_clip(current)
        deltas = []
        for i in range(4, 8):
            original, edited = a[i]['target'], b[i]['target']
            value = None if original is None or edited is None else (edited['box'][0]+edited['box'][2]-original['box'][0]-original['box'][2])/2
            deltas.append(value)
        provenance = {'type': 'lossless_png_detector_measurement', 'detector': copy.deepcopy(self.detector.provenance),
                      'reference': asdict(reference), 'current': asdict(current), 'reference_detections': a, 'current_detections': b,
                      'late_indices': [4, 5, 6, 7], 'missing_policy': 'no_imputation_no_partial_mean',
                      'reference_generation_cost_must_be_counted': True,
                      'limits': 'Condition-guided unique detection; generated actor identity and physical trajectory are not ground truth.'}
        return NativePhaseMeasurement(current.scene_token, current.instance_token, current.seed_offset, current.arm, tuple(deltas), provenance, synthetic=False)
