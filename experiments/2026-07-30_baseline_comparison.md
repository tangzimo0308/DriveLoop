# Four-strategy comparison: open / best-of-4 / text-only / DriveLoop (2026-07-30)

## Protocol
Identical evaluator, windows, device; equal render budget where
applicable (T=4). Best-of-4 = attempt 0 + plain reseeds at banks 6/7/8,
keep max (no feedback). Text-only = T=4 loop with
DRIVELOOP_TEXT_ONLY_REFINER=1 (prompt feedback, no structural rungs).

## Six-window pool (S_perc; gate = joint detection+direction passes)
| | open | bo4 | text | ours |
|---|---|---|---|---|
| mean | 0.134 | 0.188 | 0.173 | 0.304 |
| gate passes | 0 | 0 (bus 0.654 dir=0) | 1 (bus 0.672 dir=1) | 2 |
Notable: rain truck 0/0/0/0.696 (only the structural rung recovers);
bo4 wins truck-day 0.471 vs 0.439 (direction unmeasured in both).

## Motorcycle family (per-arm mean over 5 requests)
open 0.208 / bo4 0.308 / text 0.268 / ours 0.533 (grand mean);
ours best on all 7 arms; ours vs bo4 head-to-head: 31 win / 2 tie /
2 loss over 35 cases.

## Reading
1. At the same budget, WHAT is regenerated matters more than HOW
   often: structure+feedback nearly doubles equal-budget resampling on
   the main family.
2. Mechanism separation: direction repair needs feedback (text-only
   suffices on the bus); floor recovery needs the structural rung
   (rain truck); resampling provides neither.
3. Honest losses recorded (2/35 family cases; pool truck-day).
Runtime: ~3 min per attempt on the A10; 28 runs / 245 renders total.
