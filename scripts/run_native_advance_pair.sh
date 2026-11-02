#!/usr/bin/env bash
set -euo pipefail
cd /home/lesliton/projects/DriveLoop
export PATH=/home/lesliton/miniconda3/envs/drivedreamer/bin:$PATH
export PYTHONPATH="$PWD:$PWD/dreamer-datasets:$PWD/dreamer-models:$PWD/dreamer-train"
exec 9>/tmp/driveloop-m2-motion-guard.lock
flock -n 9 || { echo 'Another DriveLoop experiment is running.'; exit 1; }
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo 'GPU is in use. Wait for the current task to finish.'; exit 1
fi
out="outputs/native_advance_pair_$(date +%Y%m%d_%H%M%S)_$$"
python scripts/run_native_advance_pair.py --output-dir "$out" 2>&1 | tee "$out.log"
cp "$out.log" "$out/run.log"
tar -czf "$out.tar.gz" -C outputs "$(basename "$out")"
printf '\nDONE\nResult archive: %s.tar.gz\n' "$PWD/$out"
