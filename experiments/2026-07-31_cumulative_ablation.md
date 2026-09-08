# Cumulative module ablation (2026-07-31)

## Variants (T=4, bank0; each row adds one module)
| variant | family grand | pool mean |
|---|---|---|
| open-loop | 0.208 | 0.134 |
| +reseed (best-of-4) | 0.308 | 0.188 |
| +text (text-only loop) | 0.268 | 0.173 |
| +structural (rung-1 only, DRIVELOOP_DISABLE_SYNTHETIC_RUNG=1) | 0.302 | 0.161 |
| +synthetic (full DriveLoop) | 0.533 | 0.304 |

Rung-1-only per-arm: c162o 0.440, c162ft 0.409, c2216o 0.161,
c2216ft 0.330, c70o 0.214, c70ft 0.182, c70dims 0.380.
Pool r1: 1677 0.180, 1313 0, 2751 0.113, 1300 0, 28 0.674, 41 0.

## Reading (honest, non-monotone)
1. Synthetic rung is the decisive module (+0.231 family over r1-only).
2. Text lowers the mean vs pure reseed but is the only baseline that
   repairs bus direction (gate value, not mean value).
3. Structural strengthening is flat on means (confounded by seed
   draws) but carries real gain at matched seeds (Table seed_control).
4. Ordering matters: text must be reverted at the synthetic rung
   (cr9 ablation). Paper 5.2.4 written accordingly.
