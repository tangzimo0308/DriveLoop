# True rebinding: margin subsets make candidate_offset real (2026-07-22)

## Change
build_candidate70_runtime_subset gains --margin-windows K: the subset
now copies the bound window PLUS up to K neighbor windows per side,
taken from the source enumeration's own start groups (same scene,
front frame_idx +- k*24). Index arithmetic across cameras is invalid
in the source layout (first attempt lost CAM_BACK_LEFT and collapsed
enumeration back to one window); each group's own per-camera starts
are used instead.

## Pilot (candidate1677, K=1)
margin_shifts_admitted [-24, 0, 24]; 432 records; subset binding
candidate_start_count=3, dd2_batch_skip=1 (token match anchors the
center window), ready=True. candidate_offset is now a real source
shift: (1+off)%3 selects the -24/0/+24 window.

## Protocol constraints identified
1. Baseline subtraction must use the OFFSET's own no-injection
   baseline; per-offset baselines are rendered in the pilot.
2. Neighbor windows are not admission-checked for the bound actor;
   real-track injection may legitimately fail there (recorded via
   real_track_fallback_reason). That is part of what the pilot
   measures.
Scope: exploratory, post-v10f; paper protocol unchanged.
