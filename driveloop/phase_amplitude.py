"""Source-geometry inverse proposal, not an authorized runtime action.

Interpolated proposals MUST pass fresh exact-parameter geometry and DD2 input
validation before generation. This module never registers or renders an action.
"""
from driveloop.native_phase import number


def interpolate(x, points):
    if not points[0][0] <= x <= points[-1][0]:
        raise ValueError('Extrapolation is forbidden')
    for (a, y), (b, z) in zip(points, points[1:]):
        if a <= x <= b:
            return y + (z-y)*(x-a)/(b-a)
    raise ValueError('Invalid interpolation range')


def propose_phase(source_curve, current_phase_s, target_dx_px, late_deltas,
                  max_step_s=.125, deadband_px=1.):
    phase, target = number(current_phase_s), number(target_dx_px)
    step, deadband = number(max_step_s), number(deadband_px)
    if step <= 0 or deadband < 0:
        raise ValueError('Invalid step bound or numerical deadband')
    if len(source_curve) < 2 or any(not p['geometry_passed'] for p in source_curve):
        raise ValueError('Explicit contiguous grid of admitted source points required')
    points = [(number(p['phase_s']), number(p['source_mean_dx_px'])) for p in source_curve]
    if any(b[0] <= a[0] for a, b in zip(points, points[1:])):
        raise ValueError('Strictly increasing phase coordinates required')
    differences = [b[1]-a[1] for a,b in zip(points, points[1:])]
    if not (all(d>0 for d in differences) or all(d<0 for d in differences)):
        raise ValueError('Nonmonotone or flat projection; do not guess an inverse')
    expected = interpolate(phase, points)
    if len(late_deltas) != 4:
        raise ValueError('All four fixed late frames required')
    values = [None if x is None else number(x) for x in late_deltas]
    result = {'current_phase_s': phase, 'proposed_phase_s': phase, 'target_dx_px': target,
              'source_expected_dx_px': expected, 'measurement_dx_px': None, 'residual_px': None,
              'runtime_authorized': False, 'required_checks': ['exact_parameter_geometry', 'exact_parameter_DD2_input_preflight'],
              'limits': 'A source-only piecewise-linear inverse does not establish generated-video response.'}
    if any(x is None for x in values):
        return {**result, 'status': 'hold_unmeasured'}
    observed = sum(values)/4; residual = target-observed
    result.update(measurement_dx_px=observed, residual_px=residual)
    if abs(residual) <= deadband:
        return {**result, 'status': 'hold_within_software_deadband'}
    desired = expected+residual
    lo, hi = max(points[0][0],phase-step), min(points[-1][0],phase+step)
    ylo, yhi = interpolate(lo,points), interpolate(hi,points)
    clipped = min(max(desired,min(ylo,yhi)),max(ylo,yhi))
    inverse = sorted((y,x) for x,y in points)
    proposal = interpolate(clipped,inverse)
    proposal = min(hi,max(lo,proposal))
    return {**result, 'status':'proposal_requires_fresh_validation', 'desired_source_dx_px':desired,
            'bounded_source_dx_px':clipped, 'saturated': clipped != desired,
            'proposed_phase_s': proposal, 'step_s': proposal-phase,
            'source_predicted_residual_px': desired-interpolate(proposal,points)}
