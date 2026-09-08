# Direction probe: scoring-vocabulary alignment (2026-07-22)

## Change
_maneuver_direction_check now filters detection centers with the same
vocabulary the scorer uses: labels are normalized (person->pedestrian,
bike/cyclist->bicycle) and widened by the v10 super-class expansion via
a _direction_label_set hook (base class: normalized category; v10
subclass: expand_labels). The replaced raw-label exact match (a) made
direction silently unmeasurable for pedestrian targets (YOLO emits
"person") and (b) was stricter than the score itself.

## Offline re-probe of v10f pool bests (stored videos, yolov8x, conf 0.25)
| run | S_old | S_new | dir | delta_x |
|---|---|---|---|---|
| truck day 1677 | 0.439 | 0.439 | - | - |
| truck night 1313 | 0.000 | 0.000 | - | - |
| truck rain 2751 | 0.696 | 0.696 | 1 | +15.4 |
| moto night 1300 | 0.000 | 0.000 | - | - |
| bus 28 | 0.687 | 0.687 | 1 | +21.3 |
| bicycle 41 | 0.000 | 0.000 | - | - |

## Reading
1. Regression guard holds: S_perc reproduces bit-identically on 6/6
   runs; the vocabulary touches direction only.
2. The truck-day n/m is real detection sparsity in the selected view,
   not a vocabulary artifact; Table 5 stands unchanged. Widening the
   direction vocabulary beyond the scoring super-class (e.g., truck ->
   car) is deliberately NOT done: direction evidence must not be
   looser than scoring evidence.
3. The pedestrian fix matters for the not-yet-closed-loop candidate1409
   window: under the old probe its direction could never be measured.

## Next
True rebinding via margin subsets (correction record 1 commitment),
then FVD.
