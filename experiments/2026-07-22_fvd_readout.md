# FVD: closed-loop bests are closer to real than single passes (2026-07-22)

## Protocol
FVD-8(dup16): cam_front crops of the generated row (the same geometry
the evaluator scores), 8 frames duplicated to 16, I3D torchscript
features (stylegan-v weights), Frechet distance via scipy sqrtm.
REAL = 128 real cam_front windows sampled from the train enumeration
(seed 0, raw nuScenes jpgs resized to 448x256). OPEN = the 41
attempt-0 videos of the v10f pool + seven-arm runs; CLOSED = the 41
best-attempt videos of the same runs.

## Readout
| pair | FVD |
|---|---|
| real vs open | 2103.5 |
| real vs closed | 1751.7 |
| open vs closed | 555.9 |

## Reading
Closing the loop does not trade realism for perception score: the
returned bests are distributionally CLOSER to real nuScenes than the
single passes (-17% FVD), despite the synthetic-trajectory rung.
Claim discipline: absolute values are inflated by small generated N
(41 per set) and the nonstandard 8-frame protocol; only the relative
open-vs-closed comparison under identical N/windows is claimed.
Weights: /data/projects/DriveLoop/pretrained_models/i3d_torchscript.pt
(sha check via file size 51235320). Report: outputs/driveloop/fvd_report.json.
