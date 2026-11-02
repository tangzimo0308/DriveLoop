#!/usr/bin/env bash
# Longer user-run input audit. No diffusion/video generation.
set -euo pipefail
cd /home/lesliton/projects/DriveLoop
EVAL_PY=/home/lesliton/miniconda3/envs/drivedreamer/bin/python
EVAL_OUT=outputs/revision_20261022_eval_runtime
if [[ "${1:-}" == "--check-only" ]]; then
  exec "$EVAL_PY" scripts/audit_eval_runtime_v2.py --max-new 0
fi
if [[ $# -ne 0 ]]; then
  echo 'Usage: bash scripts/run_eval_loader_audit_20261022.sh [--check-only]' >&2
  exit 2
fi
# No automatic retry of interrupted slots, no changing source, seed or parameters.
set +e
"$EVAL_PY" scripts/audit_eval_runtime_v2.py
EVAL_EXIT=$?
set -e
"$EVAL_PY" - <<'PY'
import tarfile
from pathlib import Path
root=Path('outputs/revision_20261022_eval_runtime')
archive=root/'loader_audit_evidence.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.suffix in ('.json','.jsonl','.log') and 'dataset' not in p.relative_to(root).parts:
            tar.add(p,arcname=str(p.relative_to(root)))
print('Evidence archive:',archive.resolve(),flush=True)
PY
if [[ "$EVAL_EXIT" -ne 0 ]]; then
  echo "INPUT_AUDIT_STOPPED: exit $EVAL_EXIT; evidence preserved. Send the final log; do not change seeds or sources."
  exit "$EVAL_EXIT"
fi
echo 'DONE: input audits complete; no videos generated. Formal generation launch still requires review of these audits and final runtime freeze.'
