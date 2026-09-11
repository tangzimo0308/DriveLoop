"""Offline direction re-probe of stored v10f pool bests under the
normalized/super-class direction vocabulary. Re-detects the stored
videos; S_perc must reproduce the stored value (regression guard),
only maneuver_* metrics may change."""
import glob, json, os

RUNS = [
    ("candidate1677_truck_cut_in_v10f", "candidate1677"),
    ("candidate1313_night_truck_v10f", "candidate1313"),
    ("candidate2751_rain_truck_v10f", "candidate2751"),
    ("candidate1300_night_cut_in_v10f", "candidate1300"),
    ("candidate28_bus_v10f", "candidate28"),
    ("candidate41_bicycle_v10f", "candidate41"),
]


def main():
    from driveloop.perception_video import UltralyticsYOLODetector
    from driveloop.perception_v10 import ManeuverViewRestrictedSuperclassEvaluator
    from driveloop.schema import Generation

    print("%-34s %7s %7s %5s %9s" % ("run", "S_old", "S_new", "dir", "delta_x"))
    for run, cand in RUNS:
        g = glob.glob("outputs/driveloop/%s/*/result.json" % run)
        if not g:
            print("%-34s (missing result)" % run)
            continue
        res = json.load(open(g[0]))
        best = res.get("best_generation") or {}
        stored = ((res.get("best_evaluation") or {}).get("metrics")) or {}
        video = (best.get("artifacts") or {}).get("video")
        bl = sorted(glob.glob(
            "outputs/driveloop/%s_baseline_official/*/artifacts/iteration_00.mp4" % cand))
        if not video or not os.path.exists(str(video)) or not bl:
            print("%-34s MISSING video=%s baseline=%s" % (run, video, bool(bl)))
            continue
        ev = ManeuverViewRestrictedSuperclassEvaluator(
            detector=UltralyticsYOLODetector("yolov8x.pt", confidence_threshold=0.25),
            confidence_threshold=0.25,
            baseline_video=bl[0],
        )
        gen = Generation(
            iteration=int(best.get("iteration") or 0),
            prompt=str(best.get("prompt") or ""),
            artifacts=dict(best.get("artifacts") or {}),
            metadata=dict(best.get("metadata") or {}),
        )
        out = ev.evaluate(gen)
        m = out.metrics
        dr = m.get("maneuver_direction_consistent")
        dx = m.get("maneuver_pixel_delta_x")
        print("%-34s %7.3f %7.3f %5s %9s" % (
            run, float(stored.get("S_perc") or 0.0),
            float(m.get("S_perc", out.score)),
            "-" if dr is None else ("%.0f" % dr),
            ("%.1f" % dx) if isinstance(dx, (int, float)) else "-"))


main()
