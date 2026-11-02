#!/usr/bin/env bash
# Run manually on the server. No downloads or installation; historical outputs are preserved.
set -euo pipefail
cd /home/lesliton/projects/DriveLoop
export PATH=/home/lesliton/miniconda3/envs/drivedreamer/bin:$PATH
export PYTHONPATH="$PWD:$PWD/dreamer-datasets:$PWD/dreamer-models:$PWD/dreamer-train"
source_dir=outputs/revision_20261005_M2_expansion
args=(--dataset "$source_dir/candidate28/dataset" --identity "$source_dir/candidate28_identity.json" --window-manifest "$source_dir/candidate28_manifest.json" --seed-bank 0)
for required in "$source_dir/candidate28/dataset/labels/data.pkl" "$source_dir/candidate28/dataset/subset_report.json" "$source_dir/candidate28_identity.json" "$source_dir/candidate28_manifest.json" yolov8x.pt /data/models/dd2/pretrained_models/drivedreamer2_img_cond/pytorch_gligen_weights.bin; do
    test -f "$required" || { echo "Missing: $required"; exit 1; }
done
if [[ "${1:-}" == "--check" ]]; then
    python -c 'from scripts.run_m2_motion_guard import schedules; from driveloop.real_track_guard import GUARD_VERSION; assert len(schedules()["direct"]) == 4; print("Ready:", GUARD_VERSION, "No generation started.")'
    exit 0
fi
if [[ $# -ne 0 ]]; then echo "Usage: bash scripts/run_m2_motion_guard_bus.sh [--check]"; exit 2; fi
# DD2 uses a shared output path, so serialize invocations of this wrapper.
exec 9>/tmp/driveloop-m2-motion-guard.lock
flock -n 9 || { echo 'Another guarded experiment is running.'; exit 1; }
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo 'GPU has a compute process. Wait for it to finish before running this script.'; exit 1
fi
out="outputs/revision_motion_guard_bus_$(date +%Y%m%d_%H%M%S)_$$"
mkdir "$out"
exec > >(tee -a "$out/run.log") 2>&1
trap 'echo "FAILED: inspect $out/run.log; do not treat partial results as a completed experiment."' ERR
printf 'Results: %s\n' "$PWD/$out"
python scripts/run_m2_motion_guard.py "${args[@]}" --output-dir "$out" --stage prepare
for stage in baseline direct fixed full; do
    python scripts/run_m2_motion_guard.py "${args[@]}" --output-dir "$out" --stage "$stage" 2>&1 | tee "$out/$stage.log"
done
python scripts/summarize_m2_motion_guard.py "$out"
tar -czf "$out.tar.gz" -C outputs "$(basename "$out")"
printf '\nDONE\nResult archive: %s.tar.gz\nSummary: %s/summary.json\n' "$PWD/$out" "$PWD/$out"
