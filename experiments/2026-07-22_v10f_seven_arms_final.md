# Seven arms under v10f: final headline numbers (2026-07-22)

## Protocol
v10f (T=4; rung 1 text refinement + size/steps; rung 2 synthetic
close-range trajectory, category-correct, bare prompt; rung 3 synthetic
reseed). Arm->baseline pairing preserved from v10b (ft arms read
ft-rendered baselines; candidate70 single baseline). Bank0, v10b
evaluator, category preserved on every attempt (audited).

## Results (mean uplift closed best - open, 5 requests per arm)
| arm | v10b | v10f |
|---|---|---|
| c162 released | +0.032 | +0.093 |
| c162 ft+dim | +0.066 | +0.209 |
| c2216 released | +0.080 | +0.557 |
| c2216 ft+dim | +0.224 | +0.453 |
| c70 released | +0.048 | +0.481 |
| c70 ft | +0.123 | +0.220 |
| c70 released+dim | +0.111 | +0.265 |
| GRAND MEAN | +0.098 | +0.326 |

35 comparisons: 32 improve, 3 tie, 0 regress. Five arms contain
floored requests (open 0) recovered above 0.5 (max: c2216 released
m2 rainy-night 0 -> 0.735); the synthetic rung carries these.

## Status
v10f is the final paper protocol. Paper Sec. 4 rewritten (Tables 1/3/5
on v10f; Table 2/4 unchanged; correction disclosure in 4.1/4.5);
abstract/intro aligned (+0.326, 32/35, two joint-gate repairs, honest
source-condition floors). Remaining engineering backlog: direction
measurability densification, true rebinding via margin subsets, FVD.
