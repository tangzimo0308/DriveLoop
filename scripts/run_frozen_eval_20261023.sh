#!/usr/bin/env bash
set -euo pipefail
cd /home/lesliton/projects/DriveLoop
# Prefix length defaults to one PREASSIGNED scene-bank, not an extra pilot.
groups="${1:-1}"
exec /home/lesliton/miniconda3/envs/drivedreamer/bin/python -u scripts/run_frozen_eval.py run --groups "$groups"
