from scripts.run_m2_matched_pilot import PROMPT, PresetRefiner, schedules


def test_direct_control_uses_same_synthetic_recipe_on_every_seed():
    requests = schedules()['direct']
    assert len(requests) == 4
    assert all(r == requests[0] for r in requests)
    assert all(r.prompt == PROMPT for r in requests)
    assert all(r.condition['synthetic_trajectory_escalation'] for r in requests)
    assert all(r.condition['generation_escalation']['num_inf_steps'] == 50 for r in requests)


def test_fixed_control_declares_real_real_synthetic_synthetic_before_feedback():
    requests = schedules()['fixed']
    assert [bool(r.condition.get('synthetic_trajectory_escalation')) for r in requests] == [False, False, True, True]
    assert all(r.condition['generation_escalation']['num_inf_steps'] == 50 for r in requests)


def test_control_refiner_does_not_read_actual_feedback_or_incoming_request():
    requests = schedules()['fixed']
    refiner = PresetRefiner(requests)
    for planned in requests[1:]:
        actual = refiner.refine(object(), object())
        assert actual.prompt == planned.prompt
        assert actual.condition == planned.condition


def test_control_outputs_are_independent_mutable_copies():
    requests = schedules()['direct']
    refiner = PresetRefiner(requests)
    changed = refiner.refine(None, None)
    changed.condition['structural_escalation']['size_scale'] = 999
    assert refiner.refine(None, None).condition['structural_escalation']['size_scale'] == 1.5
    assert requests[0].condition['structural_escalation']['size_scale'] == 1.5
