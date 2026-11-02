# M1: distinguish ranking, acceptance and returned candidate

This revision does not replace archived A10 measurements or claim new GPU
results. Historical scripts/configurations must be recorded when replayed.

## Protocol

- `evaluation.score` ranks candidates. It is J when `use_task_utility=True`;
  `S_perc`, `S_ctrl`, and `S_intent` remain separate metrics.
- New generic default: `score_and_diagnosis`. A high score cannot override a
  failed diagnosis or an explicitly failed direction check.
- Paper joint gate: `joint_perception_direction`. Acceptance requires
  `S_perc >= perception_acceptance_threshold` AND measured direction = 1.
  Threshold 0.3 is taken from the archived 2026-07-21 quality-gate record,
  not fitted to the current reruns. Direction absent is unmeasured, not pass.
- By default the joint loop returns the first passing attempt. With
  `--fixed-budget` it evaluates the full budget (except an unscorable view)
  and returns the highest-ranked passing candidate.
- `best_generation/best_evaluation` retain the unconstrained ranking maximum.
  `returned_generation/returned_evaluation` identify the actual output.
  If no candidate passes, the ranking maximum is a flagged fallback, never
  silently an accepted video. Read `return_status` before using it as data.
- The API and experiment summary use return status, not a numeric shortcut.
- `legacy_score_or_diagnosis` is an explicit historical compatibility mode;
  it may accept direction failures and is NOT the paper's joint gate.

## CLI

Use a fresh output directory and the existing source-binding wrapper:

```text
python -m scripts.render_window_case [existing source and case arguments] \
  --use-task-utility --max-iterations 4 \
  --acceptance-policy joint_perception_direction \
  --perception-acceptance-threshold 0.3
```

For matched four-render comparisons, add `--fixed-budget` to every strategy
and evaluate returned quality and gate pass rate separately. Report actual
runtime; four renders do not imply equal compute. The old experiment
launchers do not become joint-gate experiments merely by using the patched
code: pass the explicit joint policy through their invocation or use the
wrapper above. Missing sample bindings and weights must be resolved before
rerunning the 35-case study.

## Correct theorem scope

For the nonempty set of evaluated candidates, the stored rank maximum is
nondecreasing. This gives no guarantee that S_perc, semantic validity or
the accepted output's rank never decreases: acceptance constraints can
exclude the unconstrained maximum. State first-pass inclusion and fixed
ranking function assumptions. Do not describe a fallback as gate-passing.

## Remaining review items

This patch does not validate detector proxies, physical cut-in semantics,
score normalization, independent scenes, FID/FVD, or feedback versus direct
trajectory injection. Those require the subsequent review steps.
