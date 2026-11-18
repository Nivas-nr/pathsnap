"""Align every train / student_test trip and save per-point labels.

Run:  python -m src.data.build_labels
Writes data/processed/labels/<partition>/<trip>.npz  (labels, route_pos, dist)
and results/eda/label_summary.csv.
"""
import numpy as np
import pandas as pd

from src.data.labels import MAX_LABEL_DIST_M, align_route
from src.data.loader import ROOT, list_trips, load_trip

OUT = ROOT / "data" / "processed" / "labels"


def main():
    rows = []
    for part in ["train", "student_test"]:
        (OUT / part).mkdir(parents=True, exist_ok=True)
        for tid in list_trips(part):
            trip = load_trip(tid, part)
            labels, pos, dist = align_route(trip)
            np.savez_compressed(OUT / part / f"{tid}.npz", labels=labels, route_pos=pos, dist=dist)
            ok = labels >= 0
            rows.append(dict(partition=part, trip=tid, n_gps=len(labels), n_labelled=int(ok.sum()),
                             pct_labelled=ok.mean() * 100, median_dist_m=float(np.median(dist[ok])),
                             p95_dist_m=float(np.percentile(dist[ok], 95)),
                             route_arcs_hit_pct=len(np.unique(pos)) / len(trip.route) * 100))
            print(f"{part:12s} {tid}  labelled {rows[-1]['pct_labelled']:5.1f}%")
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "eda" / "label_summary.csv", index=False)
    g = df.groupby("partition")
    print(f"\nThreshold {MAX_LABEL_DIST_M} m")
    print(g.agg(points=("n_gps", "sum"), labelled=("n_labelled", "sum"),
                median_dist=("median_dist_m", "median"), arcs_hit=("route_arcs_hit_pct", "mean")))


if __name__ == "__main__":
    main()
