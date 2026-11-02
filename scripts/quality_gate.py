"""Quality acceptance gate for DriveLoop v10b closed-loop runs.

accept = (S_perc >= tau) AND (maneuver_direction_consistent == 1.0)

Direction is UNMEASURED (not failed) when the metric is absent: the
direction probe needs >= 3 frames with a target-class detection in the
selected view (composite_perception._maneuver_direction_check).
"""
import argparse, glob, json, os
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from driveloop.acceptance import joint_gate


def gate_state(m, tau):
    return joint_gate(m, tau)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--tau", type=float, default=0.3)
    args = ap.parse_args()
    print("%-42s %7s %5s %9s %-10s %s" % (
        "run", "S_perc", "dir", "delta_x", "gate", "attempts (S_perc/dir)"))
    for rd in args.run_dirs:
        g = sorted(glob.glob(os.path.join(rd, "*", "result.json")))
        name = os.path.basename(rd.rstrip("/"))
        if not g:
            print("%-42s  (missing result.json)" % name)
            continue
        res = json.load(open(g[0]))
        best = (res.get("best_evaluation") or {}).get("metrics") or {}
        state = gate_state(best, args.tau)
        atts, gate_best = [], None
        for i, att in enumerate(res.get("attempt_history") or []):
            m = (att.get("evaluation") or {}).get("metrics") or {}
            sp = float(m.get("S_perc") or 0.0)
            dr = m.get("maneuver_direction_consistent")
            drs = "-" if dr is None else ("1" if float(dr) == 1.0 else "0")
            atts.append("a%d:%.3f/%s" % (i, sp, drs))
            if gate_state(m, args.tau) == "PASS" and (
                    gate_best is None or sp > gate_best[1]):
                gate_best = (i, sp)
        dx = best.get("maneuver_pixel_delta_x")
        dxs = ("%.1f" % dx) if isinstance(dx, (int, float)) else "-"
        dr = best.get("maneuver_direction_consistent")
        drs = "-" if dr is None else str(dr)
        note = " ".join(atts)
        if state != "PASS" and gate_best is not None:
            note += "  [gate-best: attempt %d S_perc=%.3f]" % gate_best
        print("%-42s %7.3f %5s %9s %-10s %s" % (
            name, float(best.get("S_perc") or 0.0), drs, dxs, state, note))


if __name__ == "__main__":
    main()
