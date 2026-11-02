from dataclasses import replace
from pathlib import Path
import pytest
from PIL import Image
from driveloop.native_phase import digest, NativePhaseEvaluator
from driveloop.native_phase_png import PNGFrame, PNGClip, PNGPhaseMeasurement, associate
from driveloop.schema import Generation

class Detector:
    provenance = {'test_double': True}
    def __call__(self, im):
        dx = im.getpixel((0, 0))[0]
        return [{'label': 'bus', 'score': .9, 'box': [10+dx, 20, 60+dx, 100]}]

@pytest.fixture
def setup(tmp_path):
    evidence = tmp_path/'evidence.json'; evidence.write_text('{}')
    clips = []
    for arm, dx in [('original', 0), ('advanced', 4)]:
        frames = []
        for i in range(8):
            path = tmp_path/f'{arm}_{i}.png'; Image.new('RGB', (448, 256), (dx, i, 0)).save(path)
            frames.append(PNGFrame(i, str(path), digest(path), (10+dx, 20, 60+dx, 100)))
        clips.append(PNGClip('scene', 'bus', 'registry', 0, arm, 'noise', tuple(frames), str(evidence), digest(evidence)))
    q = {'scene_token': 'scene', 'instance_token': 'bus', 'registry_sha256': 'registry', 'arm': 'advanced', 'target_dx_px': 4, 'tolerance_px': 1}
    g = Generation(0, '', metadata={'native_phase_contract': q})
    return clips[0], clips[1], g

def bridge(original, current, detector=None):
    return PNGPhaseMeasurement(lambda g: current, lambda g: original, detector or Detector())

def test_real_values_flow_to_evaluator(setup):
    a, b, g = setup; cb = bridge(a, b)
    result = NativePhaseEvaluator(cb).evaluate(g)
    assert result.diagnosis.passed
    assert result.metrics['native_phase_feedback']['late_deltas'] == [4]*4
    assert not result.metrics['native_phase_feedback']['synthetic']
    before = cb.detector_calls; cb(g); assert cb.detector_calls == before

@pytest.mark.parametrize('field,value', [('seed_offset', 1), ('scene_token', 'wrong'), ('instance_token', 'wrong'), ('registry_sha256', 'wrong'), ('arm', 'delayed'), ('initial_noise_sha256', 'wrong')])
def test_wrong_reference_rejected(setup, field, value):
    a, b, g = setup
    with pytest.raises(ValueError): bridge(replace(a, **{field: value}), b)(g)

def test_tampered_png_rejected_even_after_cache(setup):
    a, b, g = setup; cb = bridge(a, b); cb(g)
    Path(b.frames[6].path).write_bytes(b'changed')
    with pytest.raises(ValueError): cb(g)

def test_evidence_tamper(setup):
    a, b, g = setup; Path(a.evidence_path).write_text('changed')
    with pytest.raises(ValueError): bridge(a, b)(g)

@pytest.mark.parametrize('indices', [(0, 1, 2, 3, 4, 5, 7), (0, 1, 2, 3, 4, 5, 7, 6), (0, 1, 2, 3, 4, 5, 6, 6)])
def test_missing_reordered_duplicate_frames_rejected(setup, indices):
    a, b, g = setup
    with pytest.raises(ValueError): bridge(a, replace(b, frames=tuple(b.frames[i] for i in indices)))(g)

@pytest.mark.parametrize('ambiguous', [False, True])
def test_missing_or_ambiguous_late_frame_not_zero(setup, ambiguous):
    a, b, g = setup
    class BadDetector(Detector):
        def __call__(self, im):
            r = super().__call__(im)
            return (r*2 if ambiguous else []) if im.getpixel((0, 0)) == (4, 6, 0) else r
    result = NativePhaseEvaluator(bridge(a, b, BadDetector())).evaluate(g)
    f = result.metrics['native_phase_feedback']
    assert f['late_deltas'] == [4, 4, None, 4]
    assert f['observed_dx_px'] is None and f['error_px'] is None
    assert result.metrics['native_phase_unmeasurable'] and not result.diagnosis.passed

def test_early_absence_does_not_select_late_window(setup):
    a, b, g = setup
    class EarlyMissing(Detector):
        def __call__(self, im):
            return [] if im.getpixel((0, 0))[1] == 1 else super().__call__(im)
    assert bridge(a, b, EarlyMissing())(g).late_deltas == (4, 4, 4, 4)

def test_no_highest_score_fallback():
    raw = [{'label':'bus','score':s,'box':[0,0,10,10]} for s in (.6,.99)]
    assert associate(raw, [0,0,10,10]) == {'association_count':2,'target':None}

@pytest.mark.parametrize('size,mode', [((448,255),'RGB'), ((448,256),'L')])
def test_no_implicit_resize_or_color_conversion(setup, size, mode):
    a,b,g=setup; f=b.frames[7];Image.new(mode,size).save(f.path)
    frames=b.frames[:7]+(replace(f,sha256=digest(f.path)),)
    with pytest.raises(ValueError):bridge(a,replace(b,frames=frames))(g)

def test_same_reference_missing_still_unmeasurable(setup):
    a,b,g=setup; g=replace(g, metadata={'native_phase_contract':{**g.metadata['native_phase_contract'],'arm':'original'}})
    class Empty(Detector):
        def __call__(self, im):return []
    assert bridge(a,a,Empty())(g).late_deltas == (None,)*4
