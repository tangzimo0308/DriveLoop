from scripts.run_m2_window_control import PresetRefiner, schedules


def test_bus_and_bicycle_controls_preserve_category_and_intent():
    for category in ['bus','bicycle']:
        prompt=f'urban street, a {category} cuts in from the left toward the ego vehicle'
        plan=schedules(prompt)
        assert plan['full'][0].prompt==prompt
        assert [bool(r.condition.get('synthetic_trajectory_escalation')) for r in plan['fixed']]==[False,False,True,True]
        assert all(r.prompt==prompt for r in plan['direct'])
        assert all(r.condition['generation_escalation']['num_inf_steps']==50 for r in plan['fixed'])
        r=PresetRefiner(plan['fixed'])
        for expected in plan['fixed'][1:]:
            got=r.refine(object(),object())
            assert got.prompt==expected.prompt and got.condition==expected.condition
        distance=plan['direct'][0].condition['structural_escalation'].get('longitudinal_base_m')
        assert distance == (9.0 if category=='bicycle' else None)
