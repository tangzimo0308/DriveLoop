"""Candidate ablation adapter; not part of the completed frozen experiment.

Generate the reference refiner's complete action first, then remove exactly
one configured structural output. Prompt-reset behavior is retained. This is
a policy intervention, not a claim that future adaptive trajectories coincide.
No generation, scoring, or environment mutation occurs in this module.
"""
import copy
import os
from dataclasses import replace

MODES = ('full', 'without_synthetic', 'without_size')

class IsolatedRefiner:
    def __init__(self, reference, mode):
        if mode not in MODES:
            raise ValueError('Unknown ablation mode: ' + str(mode))
        self.reference = reference
        self.mode = mode

    def refine(self, request, evaluation):
        # These existing flags suppress multiple actions; fail rather than
        # silently combine the requested intervention with another ablation.
        if any(os.environ.get(k) == '1' for k in
               ('DRIVELOOP_TEXT_ONLY_REFINER', 'DRIVELOOP_DISABLE_SYNTHETIC_RUNG')):
            raise ValueError('Conflicting legacy refiner ablation flag')
        if not self.reference.STRUCTURAL_ESCALATION_ENABLED:
            raise ValueError('Reference structural policy must be enabled')
        base = self.reference.refine(request, evaluation)
        if self.mode == 'full':
            return base
        condition = copy.deepcopy(base.condition)
        if self.mode == 'without_synthetic':
            condition.pop('synthetic_trajectory_escalation', None)
        elif isinstance(condition.get('structural_escalation'), dict):
            condition['structural_escalation']['size_scale'] = 1.0
        # Refiner notes describe reference proposals; the appended marker makes
        # the intervention explicit. The runtime preflight must inspect the
        # actual condition and input tensors, not infer control from notes.
        return replace(base, condition=condition,
                       notes=base.notes + ['ablation_applied:' + self.mode])
