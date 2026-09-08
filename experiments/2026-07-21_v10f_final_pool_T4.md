# v10f: final pool protocol, T=4 with reseeded synthetic rung (2026-07-21)

## Protocol
T=4: a0 base (real-track), a1 text refinement (rung 1), a2 synthetic
rung (bare prompt, close range for small actors), a3 synthetic reseed.
No code change over v10e (--max-iterations 4). Bank0, v10b evaluator.

## Results (open = a0, closed = best)
| window | open | closed | dir | gate |
|---|---|---|---|---|
| truck day (1677) | 0.171 | 0.439 (a3) | unmeasured | +0.268, human review |
| truck night (1313) | 0.000 | 0.000 | - | floor |
| truck rain (2751) | 0.000 | 0.696 (a3) | 1.0 | PASS |
| moto night (1300) | 0.000 | 0.000 | - | floor in-loop (offline 4/4 seeds recover; see cr9 record) |
| bus day (28) | 0.634 dir=0 | 0.687 (a2) | 1.0 | PASS, direction repaired |
| bicycle day (41) | 0.000 | 0.000 | - | floor |

## Reading
1. Reseeding the synthetic rung pays twice on the rain window: a2
   recovers detection with the WRONG direction (0.492/dir=0), a3
   recovers more with the RIGHT direction (0.696/dir=1) and passes the
   joint gate. Keep-best discards the bus a3 regression (0.606 < 0.687).
2. Two joint-gate passes in the pool (rain truck recovery, bus
   direction repair); closed >= open everywhere.
3. Floors: truck night, bicycle day (all synthetic distances and seeds
   tested so far fail); moto night is seed-limited in-loop.

## Status
v10f is the final paper protocol for the pool. Seven-arm family re-run
under v10f pending (consistency of Table 1); paper tables to be
finalized after it lands.
