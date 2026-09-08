# v10e pool: synthetic rung with bare prompt + close range (2026-07-21)

## Results (best-by-S_perc; attempts a0/a1/a2, path REAL/SYN)
| window | v10d best | v10e best | note |
|---|---|---|---|
| truck day (1677) | 0.171 | 0.193 | a2 SYN bare-prompt now best; closed>open +0.022 |
| truck night (1313) | 0.000 | 0.000 | floor |
| truck rain (2751) | 0.181 | 0.492 dir=0 | detection up, direction measured WRONG -> gate DIR_FAIL |
| moto night (1300) | 0.000 | 0.000 | a2 seed (offset 2) drew zero; cr9 ablation shows 4/4 other seeds recover (0.143-0.650) |
| bus day (28) | 0.655 | 0.687 dir=1 | gate PASS; closed>open +0.053 |
| bicycle day (41) | 0.000 | 0.000 | floor |

## Reading
1. Bare-prompt synthetic rung helps wherever the rung fires usefully
   (truck day, rain detection, bus). Direction on the rain window is
   measured inconsistent at 20 m: detection recovery and maneuver
   direction remain separable failure modes, now demonstrated on the
   synthetic path too (eyeball via contact sheet pending).
2. moto night is seed-limited, not condition-limited: a single synthetic
   attempt gives one draw; the ablation shows most draws recover.

## Decision (user-approved)
T=4: give the synthetic rung one reseeded retry (attempt 3 = level 3,
seed offset 3). No code change (escalation ladder already supports it);
--max-iterations 4. Protocol tag v10f = final paper protocol; pool
re-run first, then the seven-arm family for full consistency.
