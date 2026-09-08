# True-rebinding pilot readout (candidate1677, K=1) (2026-07-22)

## Protocol
Per offset {0,1,2}: no-injection baseline of THAT window (honest
subtraction), then a single-pass REAL-path probe at the attempt-0
recipe with condition source_rebinding. Bank0, v10b evaluator.

## Readout
| off | skip | front frame_idx | S_perc | det | path | fallback |
|---|---|---|---|---|---|---|
| 0 | 1 | 120 | 0.171 | 1 | REAL | none |
| 1 | 2 | 144 | 0.000 | 0 | REAL | none |
| 2 | 0 | 96  | 0.000 | 0 | REAL | none |

## Findings
1. Mechanism proven: candidate_offset now binds genuinely different
   source windows (three distinct front frames), for the first time in
   the project's history.
2. Regression anchor: offset 0 reproduces the original open-loop
   0.171 exactly; the margin subset does not perturb the center window.
3. Science: on this window, true rebinding does not recover anything.
   Real-track injection survives in both neighbors (the instance is
   still present) but is undetectable there (det=0). Window admission
   already selected the best real window; neighbors are strictly worse.
   The synthetic rung (0.439 on this window under v10f) remains the
   effective recovery mechanism.

## Status
The correction-record-1 commitment is closed with a measured answer:
rebinding was a no-op, is now real, and buys nothing on the pilot.
Margin infrastructure is retained for future condition-aware source
search. Optional follow-up (not scheduled): escalated-recipe probes on
neighbor windows, more pilot windows. Next backlog item: FVD.
