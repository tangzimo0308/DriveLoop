# Synthetic-rung distance sweep on floored windows (2026-07-21)

## Protocol
Single-pass renders, synthetic rung requested via condition, escalation
recipe fixed (size 1.5, steps 50, lateral_base_m 3.5), sweeping
longitudinal_base_m over 9/12/15/20 m. Zero code change (the surface
plan honors absolute escalation overrides). Seed bank 0, attempt 0.

## Readout (S_perc / dir / det)
| window | 9 m | 12 m | 15 m | 20 m |
|---|---|---|---|---|
| truck night (1313) | 0 | 0 | 0 | 0 |
| truck rain (2751) | 0 | 0 | 0 | 0.679 dir=1 det=4 |
| bicycle day (41) | 0 | 0 | 0 | 0 |
| moto night (1300) | 0.650 det=5 | 0.547 det=2 | 0 | 0 |

## Findings
1. The optimal synthetic distance is category-size dependent: the night
   motorcycle becomes detectable only at close range (9-12 m); the rain
   truck recovers only at the default 20 m (close range dies in rain).
2. The "small + degraded is unrecoverable" cell falls: night motorcycle
   0 -> 0.650 at 9 m (direction unmeasured, routes to human review).
   Rain truck reaches a full joint-gate pass (0.679, dir=1, dx=+12).
3. True floors within the tested grid: truck night and bicycle day
   (zero at every distance).

## Decision
Refiner rung-2 gains a small-actor close-range override
(longitudinal_base_m = 9.0 for motorcycle/bicycle/pedestrian; large
actors keep side defaults). Protocol tag v10d; pool re-run follows.
Note: the motorcycle seven-arm family used the pre-synthetic-rung
protocol at attempt 2; a consistency re-run under v10d is pending and
its numbers may improve.
