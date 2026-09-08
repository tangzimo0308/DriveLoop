# cr9 ablation: refined-prompt additions suppress the synthetic actor (2026-07-21)

## Question
The 9 m synthetic rung recovers the night motorcycle in the sweep
(S_perc 0.650, seed 0, bare prompt) but scores 0.000 at loop attempt 2
(seed offset 2, additions-refined prompt). Seed or prompt?

## Ablation (fixed 9 m synthetic rung, night motorcycle window)
| job | prompt | seed bank | S_perc | det |
|---|---|---|---|---|
| sweep d9 | bare | 0 | 0.650 | 5 |
| bank1_bare | bare | 1 | 0.395 | 2 |
| bank2_bare | bare | 2 | 0.332 | 3 |
| bank3_bare | bare | 3 | 0.143 | 1 |
| bank0_loopprompt | loop attempt-2 refined | 0 | 0.000 | 0 |

## Verdict
Prompt, not seed. The bare prompt recovers detections at 4/4 seeds
(3/4 above tau=0.3); the additions-refined prompt kills the recovery at
the seed where the bare prompt scores highest. Text amplification
("remains large, visible, unoccluded", "high contrast", etc.) steers
DD2 away from realizing the close synthetic actor at night.

## Fix
At the synthetic rung (escalation level >= 2) the refiner reverts to
the ORIGINAL user prompt (preserved in
condition["driveloop_original_prompt"]) and relies on the structural
condition alone. Prompt-text refinement remains the rung-1 lever.
Protocol tag v10e; pool re-run follows. PERCEPTION_ESCALATION prompt
ladder remains reachable in the escalation-disabled ablation arm.
