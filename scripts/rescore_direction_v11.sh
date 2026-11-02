#!/usr/bin/env bash
set -euo pipefail
cd /home/lesliton/projects/DriveLoop
export PATH=/home/lesliton/miniconda3/envs/drivedreamer/bin:$PATH
export PYTHONPATH="$PWD:$PWD/dreamer-datasets:$PWD/dreamer-models:$PWD/dreamer-train"
exec 9>/tmp/driveloop-m2-motion-guard.lock
flock -n 9 || { echo 'Another DriveLoop task is using the GPU.'; exit 1; }
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo 'GPU is in use; wait for other work to finish.'; exit 1
fi
out="outputs/direction_v11_rescore_$(date +%Y%m%d_%H%M%S)_$$"
python scripts/rescore_direction_v11.py --output-dir "$out" 2>&1 | tee "$out.log"
cp "$out.log" "$out/run.log"
tar -czf "$out.tar.gz" -C outputs "$(basename "$out")"
printf '\nDONE\nResult archive: %s.tar.gz\n' "$PWD/$out"
