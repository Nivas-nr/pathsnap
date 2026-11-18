"""Dataset exploration: per-trip inventory, data-quality checks and summary plots.

Run:  python -m src.data.eda
Outputs go to results/eda/ (CSV) and reports/figures/eda/ (PNG).
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from src.data.loader import ROOT, LocalProjection, list_trips, load_trip
from src.features.geometry import point_segment_distance

OUT_CSV = ROOT / "results" / "eda"
OUT_FIG = ROOT / "reports" / "figures" / "eda"
PARTITIONS = ["train", "student_test"]


def gps_to_route_distance(trip, proj):
    """For each GPS point, distance (m) to the nearest arc of the reference route."""
    nx, ny = proj.to_xy(trip.nodes["lon"].to_numpy(), trip.nodes["lat"].to_numpy())
    r = np.unique(trip.route)
    u, v = trip.arcs["u"].to_numpy()[r], trip.arcs["v"].to_numpy()[r]
    px, py = proj.to_xy(trip.track["lon"].to_numpy(), trip.track["lat"].to_numpy())
    best = np.full(len(px), np.inf)
    for s in range(0, len(r), 500):  # chunk to bound memory
        d, _ = point_segment_distance(px[:, None], py[:, None], nx[u[s:s+500]], ny[u[s:s+500]],
                                      nx[v[s:s+500]], ny[v[s:s+500]])
        best = np.minimum(best, d.min(axis=1))
    return best


def inspect_trip(trip_id, partition):
    trip = load_trip(trip_id, partition)
    tr, nodes, arcs, route = trip.track, trip.nodes, trip.arcs, trip.route
    proj = LocalProjection.for_track(tr)
    x, y = proj.to_xy(tr["lon"].to_numpy(), tr["lat"].to_numpy())
    dt = np.diff(tr["t"].to_numpy())
    step = np.hypot(np.diff(x), np.diff(y))
    speed = np.divide(step, dt, out=np.full_like(step, np.nan), where=dt > 0)

    n_nodes, n_arcs = len(nodes), len(arcs)
    u, v = arcs["u"].to_numpy(), arcs["v"].to_numpy()
    g = coo_matrix((np.ones(n_arcs), (u, v)), shape=(n_nodes, n_nodes))
    n_weak, weak_lbl = connected_components(g, directed=True, connection="weak")
    n_strong, _ = connected_components(g, directed=True, connection="strong")
    degree = np.bincount(np.concatenate([u, v]), minlength=n_nodes)
    # arcs that appear in both directions (two-way streets)
    pairs = set(zip(u.tolist(), v.tolist()))
    two_way = sum((b, a) in pairs for a, b in pairs) / max(len(pairs), 1)

    row = dict(
        partition=partition, trip=trip_id,
        n_gps=len(tr), duration_s=tr["t"].iloc[-1] - tr["t"].iloc[0],
        n_nan=int(tr.isna().sum().sum()),
        n_dup_points=int(tr.duplicated(["lon", "lat"]).sum()),
        t_non_increasing=int((dt <= 0).sum()),
        pct_dt_1s=float(np.mean(dt == 1) * 100),
        n_gaps_gt_5s=int((dt > 5).sum()), max_gap_s=float(dt.max()),
        extent_x_km=float(np.ptp(x) / 1000), extent_y_km=float(np.ptp(y) / 1000),
        path_len_km=float(step.sum() / 1000),
        median_speed_ms=float(np.nanmedian(speed)), max_speed_ms=float(np.nanmax(speed)),
        n_speed_gt_50ms=int(np.nansum(speed > 50)),
        n_nodes=n_nodes, n_arcs=n_arcs, n_weak_components=n_weak, n_strong_components=n_strong,
        n_isolated_nodes=int((degree == 0).sum()), n_self_loops=int((u == v).sum()),
        pct_two_way=float(two_way * 100),
        largest_weak_comp_pct=float(np.bincount(weak_lbl).max() / n_nodes * 100),
    )
    if route is not None:
        valid = (route >= 0) & (route < n_arcs)
        rr = route[valid]
        row.update(
            route_len=len(route), route_invalid_ids=int((~valid).sum()),
            route_unique_arcs=len(np.unique(rr)),
            route_consecutive_repeats=int((np.diff(rr) == 0).sum()),
            # continuity: end node of arc i must be start node of arc i+1
            route_breaks=int((v[rr[:-1]] != u[rr[1:]]).sum()),
        )
        d = gps_to_route_distance(trip, proj)
        row.update(gps_route_dist_median_m=float(np.median(d)), gps_route_dist_p95_m=float(np.percentile(d, 95)),
                   gps_route_dist_max_m=float(d.max()), pct_gps_within_20m=float(np.mean(d <= 20) * 100))
    return row, dt


def main():
    OUT_CSV.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    rows, all_dt = [], {p: [] for p in PARTITIONS}
    for p in PARTITIONS:
        for tid in list_trips(p):
            row, dt = inspect_trip(tid, p)
            rows.append(row)
            all_dt[p].append(dt)
            print(f"{p:12s} {tid}  gps={row['n_gps']:6d}  arcs={row['n_arcs']:6d}  breaks={row.get('route_breaks')}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV / "trip_inventory.csv", index=False)

    summary = df.groupby("partition").agg(
        trips=("trip", "count"), gps_total=("n_gps", "sum"), gps_min=("n_gps", "min"),
        gps_median=("n_gps", "median"), gps_max=("n_gps", "max"),
        nan_total=("n_nan", "sum"), gaps_gt_5s=("n_gaps_gt_5s", "sum"), max_gap_s=("max_gap_s", "max"),
        trips_with_route_breaks=("route_breaks", lambda s: int((s > 0).sum())),
        gps_route_dist_median_m=("gps_route_dist_median_m", "median"),
        pct_gps_within_20m=("pct_gps_within_20m", "mean"),
    )
    summary.to_csv(OUT_CSV / "partition_summary.csv")
    print("\n", summary.T.to_string())

    # --- figures ---
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for p in PARTITIONS:
        sub = df[df.partition == p]
        ax[0].hist(sub["n_gps"], bins=20, alpha=0.6, label=p)
        dt = np.concatenate(all_dt[p])
        ax[1].hist(np.clip(dt, 0, 15), bins=np.arange(0, 16.5, 0.5), alpha=0.6, label=p, log=True)
        ax[2].hist(sub["gps_route_dist_median_m"], bins=20, alpha=0.6, label=p)
    ax[0].set(title="GPS points per trip", xlabel="points", ylabel="trips")
    ax[1].set(title="Sampling interval (clipped at 15 s)", xlabel="seconds", ylabel="count (log)")
    ax[2].set(title="Median GPS-to-route distance per trip", xlabel="metres", ylabel="trips")
    for a in ax:
        a.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIG / "dataset_overview.png", dpi=120)
    print(f"\nSaved {OUT_CSV / 'trip_inventory.csv'} and figures to {OUT_FIG}")


if __name__ == "__main__":
    main()
