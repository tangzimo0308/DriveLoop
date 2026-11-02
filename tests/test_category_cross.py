import copy
from types import SimpleNamespace
import pytest
from driveloop.schema import DriveLoopRequest
from scripts import run_category_cross as runner


def test_variants_change_only_prompt_noun_and_injected_category(monkeypatch):
    entry={'category':'bus','ego':{'center_ego':[16,3,1.5],'dims':[2.6,3.2,10],'heading_ego':.02},'frame_idx':144}
    base={'scene_description':{'value':'frozen source'},'boxes3d':{'per_frame_append_ego':[entry]},'audit':{}}
    monkeypatch.setattr(runner,'make_backend',lambda *a:SimpleNamespace(_build_override_json=lambda *a:copy.deepcopy(base)))
    prompt='a bus cuts in from the left'
    request=DriveLoopRequest(prompt='original user prompt',condition={'keep':'fixed','dd2_condition':{'text_prompt':prompt}})
    metadata={'dd2_prompt':prompt,'dd2_structural_input_plan':{},'dd2_override_candidate_plan':{}}
    variants=runner.make_variants({'binding':{}},[entry],metadata,request)
    assert [(v['prompt_category'],v['box_category']) for v in variants]==runner.FACTORS
    for v in variants:
        assert v['request']['prompt']==f"a {v['prompt_category']} cuts in from the left"
        e=v['override']['boxes3d']['per_frame_append_ego'][0]
        assert e['category']==v['box_category'] and e['ego']==entry['ego']
        assert v['override']['scene_description']==base['scene_description']
        assert v['request']['condition']['keep']=='fixed'
    assert entry['category']=='bus' and request.prompt=='original user prompt'


def matched_rows():
    rows=[]
    sig=lambda x:{'sha256':x}
    for text,box in runner.FACTORS:
        prompt=f'a {text}'
        rows.append({'prompt_category':text,'box_category':box,'request':{'condition':{'dd2_condition':{'text_prompt':prompt}}},
            'latent_audit':{'scale':1,'initial_noise':sig('noise'),'image_latent_before':sig('image'),'image_latent_after':sig('image')},
            'generation':{'metadata':{'dd2_runtime_input_audit':{'prompt_override':prompt,'prompt_embed':sig(text),'box_downsampler_input':sig(box),
                'img_cond':sig('image'),'grounding_downsampler_input':sig('map')}}}})
    return rows


def test_valid_factor_matrix():
    runner.check_runtime_pairs(matched_rows())

@pytest.mark.parametrize('key',['initial_noise','image_latent_before','prompt_embed','box_downsampler_input','img_cond','grounding_downsampler_input'])
def test_unplanned_input_changes_are_rejected(key):
    rows=matched_rows()
    target=rows[1]['latent_audit'] if key in ('initial_noise','image_latent_before') else rows[1]['generation']['metadata']['dd2_runtime_input_audit']
    target[key]={'sha256':'unexpected'}
    with pytest.raises(AssertionError):runner.check_runtime_pairs(rows)

@pytest.mark.parametrize('key',['prompt_embed','box_downsampler_input'])
def test_ignored_experimental_factor_is_rejected(key):
    rows=matched_rows()
    for row in rows:row['generation']['metadata']['dd2_runtime_input_audit'][key]={'sha256':'same-for-every-case'}
    with pytest.raises(AssertionError):runner.check_runtime_pairs(rows)
