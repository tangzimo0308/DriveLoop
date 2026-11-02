# Release scope and supplementary code

The main README retains the historical main-paper experiment summaries. Later studies belong to the appendix. This code release does not launch new experiments or change that arrangement.

## What is included

- Revised joint acceptance, separate ranking and return status, source binding, geometry checks, and detector runtime handling in `driveloop/`.
- Frozen evaluation and runtime-audit tools in `scripts/run_frozen_eval.py` and `scripts/eval_runtime_support.py`, plus development/capture scripts and CPU/mock tests.
- Exact supplementary module-ablation source snapshots in `module_ablation_20261027/`. `source_manifest.json` records original relative paths and SHA-256 hashes. These files are archival experiment code, not a standalone portable benchmark.

## Required external materials

Obtain nuScenes and the released DriveDreamer-2 weights under their respective terms. Model weights, datasets, generated videos, private human annotation files, and local execution logs are not included in this update. Configure the paths in `ENV.py` and individual launchers for your installation. Several historical and supplementary scripts retain their original absolute server paths.

The frozen evaluation also requires the original protocol/runtime records under `outputs/revision_20261022_eval_protocol` and `outputs/revision_20261022_eval_runtime`. The module snapshots originally lived in `outputs/revision_20261027_module_rollout` and depend on the matching preflight records, frozen reference results, and receipts. Their original source paths are listed in the manifest. Do not import or execute the archived launchers without inspecting these dependencies; some change the working directory at import time.

Reference freeze SHA-256: `5a4b765b63d6a606d1e3ab197060e448a39f422423abe282657a63280076989a`.
Module protocol SHA-256: `de8bed9a849a2c19d3de739aee4b90b730050a041b2f2ee73b8720fb63c7c61d`.

Checks are intended to reject mismatched inputs, code, model files, noise, or receipts. Do not disable checks or rewrite frozen hashes merely to make a different installation pass. A new installation requires a separately documented protocol and cannot be described as the original frozen run.

## Interpretation

The completed frozen study assigned 144 runs (four methods across 12 sources and three seed banks). The module study added 72 runs for two ablations and reused 36 full-method reference runs. These are supplementary evaluations. An input audit verifies runtime conditions, not video quality; completion verifies the assigned computation finished, not that the requested maneuver succeeded. Proxy acceptance and perception ranking must be reported separately from semantic success.

Some human judgments could not reliably identify the requested target vehicle, and no independent second annotator was available. Those judgments must not be presented as independently verified target identity or maneuver ground truth.

The historical summaries remain useful records, but complete original videos and attempt-level artifacts for all 41 historical cases have not been recovered. Current scripts and tests do not fill that evidence gap.

## Quick validation without generation

From the repository root, in an environment with the project dependencies:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest tests/test_joint_acceptance.py tests/test_runner_task_utility.py tests/test_frozen_eval.py
```

These tests use synthetic fixtures/mocks. Test outputs are not experiment results. GPU generation should be started explicitly only after checking the relevant protocol, inputs, budget, and output location.
