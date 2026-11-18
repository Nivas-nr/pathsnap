"""Choose candidate radius R and size K from the TRAIN partition only.

For each (R, K) we measure:
  miss rate   - % of labelled points whose true arc is not among its candidates
  avg cands   - mean candidates per point (cost for the classifier and HMM)
  empty rate  - % of points with no candidate at all

Run:  python -m src.features.candidate_study
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.loader import ROOT, LocalProjection, list_trips, load_trip
from src.features.candidates import ArcIndex

RADII = [20, 30, 40, 50, 75]
KS = [2, 3, 4, 6, 8, 12]


def main():
    lab_dir = ROOT / "data" / "processed" / "labels" / "train"
    counts = {(r, k): [0, 0, 0, 0] for r in RADII for k in KS}  # miss, labelled, cands, empty
    n_points = 0
    for tid in list_trips("train"):
        trip = load_trip(tid, "train")
        labels = np.load(lab_dir / f"{tid}.npz")["labels"]
        proj = LocalProjection.for_track(trip.track)
        px, py = proj.to_xy(trip.track["lon"].to_numpy(), trip.track["lat"].to_numpy())
        idx = ArcIndex(trip, proj)
        full = idx.candidates(px, py, max(RADII), max(KS))
        n_points += len(px)
        lab_mask = labels >= 0
        for r in RADII:
            for k in KS:
                c = full[(full.dist_m <= r)]
                c = c.assign(dr=c.groupby("point")["dist_m"].rank(method="dense") - 1)
                c = c[c.dr < k]
                hit = np.zeros(len(px), bool)
                true_rows = c[c["arc"].to_numpy() == labels[c["point"].to_numpy()]]
                hit[true_rows["point"].to_numpy()] = True
                s = counts[(r, k)]
                s[0] += int((lab_mask & ~hit).sum())
                s[1] += int(lab_mask.sum())
                s[2] += len(c)
                s[3] += int(len(px) - c["point"].nunique())
        print("done", tid)
    rows = [dict(radius_m=r, k=k, miss_rate_pct=s[0] / s[1] * 100, avg_candidates=s[2] / n_points,
                 empty_pct=s[3] / n_points * 100) for (r, k), s in counts.items()]
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "candidates"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "candidate_study_train.csv", index=False)
    print(df.pivot(index="k", columns="radius_m", values="miss_rate_pct").round(2).to_string())
    print(df.pivot(index="k", columns="radius_m", values="avg_candidates").round(1).to_string())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in RADII:
        sub = df[df.radius_m == r]
        ax.plot(sub["avg_candidates"], sub["miss_rate_pct"], "o-", label=f"R = {r} m")
        for _, row in sub.iterrows():
            ax.annotate(f"K={int(row.k)}", (row.avg_candidates, row.miss_rate_pct), fontsize=7)
    ax.set(xlabel="avg candidates per point", ylabel="candidate miss rate (%)", yscale="log",
           title="Candidate set trade-off (train partition)")
    ax.legend()
    fig.tight_layout()
    (ROOT / "reports" / "figures" / "candidates").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "reports" / "figures" / "candidates" / "candidate_tradeoff.png", dpi=120)


if __name__ == "__main__":
    main()
