"""Feature engineering: one row per (GPS point, candidate arc).

Candidate set: R = 50 m, K = 8 distinct distances (chosen in candidate_study on train).
Target `is_true` = candidate arc equals the point's aligned reference arc.
Features never look at labels, so the same code runs at inference time.

Run:  python -m src.features.build_features
Writes data/processed/features_<partition>.parquet
"""
import time

import numpy as np
import pandas as pd

from src.data.loader import ROOT, LocalProjection, Trip, list_trips, load_trip
from src.features.candidates import ArcIndex
from src.features.geometry import angle_diff_deg, bearing_deg, point_segment_distance

RADIUS_M = 50.0
K = 8
HEADING_WINDOW = 2       # heading from point i-2 to i+2
CONTEXT_OFFSETS = [-2, -1, 1, 2]
SPREAD_WINDOW = 11       # points, centred
MIN_HEADING_DISP_M = 5.0  # below this the heading is noise

FEATURES = [
    "dist_m", "dist_rank", "dist_minus_min", "dist_ratio", "t", "t_edge",
    "angle_diff", "cos_angle", "heading_ok", "ctx_dist", "ctx_minus_min",
    "speed_ms", "spread_m", "dt_prev", "dt_next", "n_cands",
    "arc_len", "is_two_way", "in_deg_u", "out_deg_v",
]


def point_features(px, py, t):
    n = len(px)
    i0 = np.clip(np.arange(n) - HEADING_WINDOW, 0, n - 1)
    i1 = np.clip(np.arange(n) + HEADING_WINDOW, 0, n - 1)
    dx, dy = px[i1] - px[i0], py[i1] - py[i0]
    disp = np.hypot(dx, dy)
    dt = np.maximum(t[i1] - t[i0], 1e-6)
    sx = pd.Series(px).rolling(SPREAD_WINDOW, center=True, min_periods=1).std().fillna(0).to_numpy()
    sy = pd.Series(py).rolling(SPREAD_WINDOW, center=True, min_periods=1).std().fillna(0).to_numpy()
    gaps = np.diff(t)
    return pd.DataFrame({
        "heading": bearing_deg(dx, dy),
        "heading_ok": (disp >= MIN_HEADING_DISP_M).astype(np.int8),
        "speed_ms": disp / dt,
        "spread_m": np.hypot(sx, sy),
        "dt_prev": np.r_[0.0, gaps],
        "dt_next": np.r_[gaps, 0.0],
    })


def trip_features(trip: Trip, labels: np.ndarray | None = None) -> pd.DataFrame:
    proj = LocalProjection.for_track(trip.track)
    px, py = proj.to_xy(trip.track["lon"].to_numpy(), trip.track["lat"].to_numpy())
    t = trip.track["t"].to_numpy()
    idx = ArcIndex(trip, proj)
    df = idx.candidates(px, py, RADIUS_M, K)
    p, a = df["point"].to_numpy(), df["arc"].to_numpy()

    pf = point_features(px, py, t)
    for c in ["heading_ok", "speed_ms", "spread_m", "dt_prev", "dt_next"]:
        df[c] = pf[c].to_numpy()[p]

    # arc geometry and topology
    u, v = idx.u, idx.v
    ax, ay, bx, by = idx.nx[u[a]], idx.ny[u[a]], idx.nx[v[a]], idx.ny[v[a]]
    df["arc_len"] = np.hypot(bx - ax, by - ay)
    n_nodes = len(idx.nx)
    in_deg = np.bincount(v, minlength=n_nodes)
    out_deg = np.bincount(u, minlength=n_nodes)
    df["in_deg_u"], df["out_deg_v"] = in_deg[u[a]], out_deg[v[a]]
    # two-way street: the reverse arc (v -> u) also exists
    keys = u.astype(np.int64) * n_nodes + v
    rev = v[a].astype(np.int64) * n_nodes + u[a]
    df["is_two_way"] = np.isin(rev, keys).astype(np.int8)

    # direction agreement; unreliable heading -> neutral 90 deg
    ang = angle_diff_deg(pf["heading"].to_numpy()[p], bearing_deg(bx - ax, by - ay))
    ok = df["heading_ok"].to_numpy() == 1
    df["angle_diff"] = np.where(ok, ang, 90.0)
    df["cos_angle"] = np.where(ok, np.cos(np.deg2rad(ang)), 0.0)

    # relative distance within the point's candidate set
    gmin = df.groupby("point")["dist_m"].transform("min")
    df["dist_minus_min"] = df["dist_m"] - gmin
    df["dist_ratio"] = (df["dist_m"] + 1.0) / (gmin + 1.0)
    df["t_edge"] = np.minimum(df["t"], 1 - df["t"])
    df["n_cands"] = df.groupby("point")["arc"].transform("size")

    # temporal context: how close are neighbouring GPS points to this same arc
    n = len(px)
    ctx = np.zeros(len(df))
    for off in CONTEXT_OFFSETS:
        j = np.clip(p + off, 0, n - 1)
        d, _ = point_segment_distance(px[j], py[j], ax, ay, bx, by)
        ctx += np.minimum(d, 2 * RADIUS_M)
    df["ctx_dist"] = ctx / len(CONTEXT_OFFSETS)
    df["ctx_minus_min"] = df["ctx_dist"] - df.groupby("point")["ctx_dist"].transform("min")

    df["trip"] = trip.trip_id
    if labels is not None:
        df["label"] = labels[p]
        df["is_true"] = (a == labels[p]).astype(np.int8)
    return df


def main():
    for part in ["train", "student_test"]:
        frames, s = [], time.time()
        for tid in list_trips(part):
            labels = np.load(ROOT / "data" / "processed" / "labels" / part / f"{tid}.npz")["labels"]
            frames.append(trip_features(load_trip(tid, part), labels))
        df = pd.concat(frames, ignore_index=True)
        out = ROOT / "data" / "processed" / f"features_{part}.parquet"
        df.to_parquet(out, index=False)
        lab = df[df.label >= 0]
        print(f"{part}: {len(df):,} rows, {df.groupby(['trip','point']).ngroups:,} points, "
              f"{lab.is_true.mean()*100:.1f}% positive (labelled rows), {time.time()-s:.0f}s -> {out.name}")


if __name__ == "__main__":
    main()
