"""Read-only audit of archived scores and existing full attempt logs. No GPU.

Example: python -m scripts.audit_m1_protocol --evidence-root ../DriveLoop
         --output /tmp/m1-audit.json
This cannot reconstruct missing direction metrics or reproduce generation.
"""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

from driveloop.acceptance import joint_gate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root, threshold):
    plot_path = root / "results/driveloop_plotdata.json"
    plot = json.loads(plot_path.read_text())
    rows = [dict(arm=arm, case=case, open=data["attempts"][0], returned=data["best"])
            for arm, cases in plot.items() if arm.startswith("cl_v10f_")
            for case, data in cases.items()]
    historical = {
        "source": str(plot_path), "sha256": digest(plot_path), "cases": rows,
        "count": len(rows), "open_mean": mean(r["open"] for r in rows),
        "returned_mean": mean(r["returned"] for r in rows),
        "improved": sum(r["returned"] > r["open"] + 1e-6 for r in rows),
        "tied": sum(abs(r["returned"] - r["open"]) <= 1e-6 for r in rows),
        "regressed": sum(r["returned"] < r["open"] - 1e-6 for r in rows),
        "limitation": "Archived score arithmetic only; no direction, utility, seed or video verification.",
    }
    runs = []
    for path in sorted((root / "outputs/driveloop").glob("*/*/result.json")):
        data = json.loads(path.read_text())
        attempts = []
        for attempt in data.get("attempt_history", []):
            e = attempt["evaluation"]
            metrics = e.get("metrics", {})
            attempts.append({"iteration": attempt["iteration"], "recorded_status": attempt["status"],
                             "rank_score": e["score"], "S_perc": metrics.get("S_perc"),
                             "direction": metrics.get("maneuver_direction_consistent"),
                             "joint_gate": joint_gate(metrics, threshold)})
        passing = [a for a in attempts if a["joint_gate"] == "PASS"]
        selected = max(passing or attempts, key=lambda a: a["rank_score"]) if attempts else None
        runs.append({"source": str(path), "sha256": digest(path), "attempts": attempts,
                     "replay_mode": "fixed_budget_rank_best_accepted_else_rank_best_unaccepted",
                     "joint_accepted": bool(passing), "replayed_return": selected,
                     "recorded_best": data.get("best_evaluation"),
                     "limitation": "Offline gate replay of recorded attempts; no new generation or independent evaluation."})
    return {"schema": "driveloop_m1_audit.v1", "threshold": threshold,
            "historical_score_table": historical, "available_full_logs": runs,
            "sources_modified": False, "gpu_used": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=.3)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; use a new path to preserve audit evidence")
    result = audit(args.evidence_root.resolve(), args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    historical = result["historical_score_table"]
    print(json.dumps({"historical_count": historical["count"],
                      "open_mean": historical["open_mean"],
                      "returned_mean": historical["returned_mean"],
                      "full_logs": len(result["available_full_logs"]),
                      "full_log_joint_passes": sum(r["joint_accepted"] for r in result["available_full_logs"])}, indent=2))


if __name__ == "__main__":
    main()
