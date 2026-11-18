"""Candidate generation: for each GPS point, the K nearest arcs within radius R.

Spatial index: arcs are line segments, but KD-trees index points. So we sample
each arc every SAMPLE_STEP_M metres and index the samples. A GPS point within R
of a segment is always within R + SAMPLE_STEP_M/2 of one of its samples, so a
ball query with that padded radius finds every arc that can qualify; the exact
point-to-segment distance then filters and ranks them.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.data.loader import LocalProjection, Trip
from src.features.geometry import point_segment_distance

SAMPLE_STEP_M = 5.0


class ArcIndex:
    def __init__(self, trip: Trip, proj: LocalProjection, bbox_pad_m: float = 200.0):
        self.nx, self.ny = proj.to_xy(trip.nodes["lon"].to_numpy(), trip.nodes["lat"].to_numpy())
        self.u, self.v = trip.arcs["u"].to_numpy(), trip.arcs["v"].to_numpy()
        px, py = proj.to_xy(trip.track["lon"].to_numpy(), trip.track["lat"].to_numpy())

        ax, ay, bx, by = self.nx[self.u], self.ny[self.u], self.nx[self.v], self.ny[self.v]
        # only index arcs that touch the area around the track
        lo_x, hi_x, lo_y, hi_y = px.min() - bbox_pad_m, px.max() + bbox_pad_m, py.min() - bbox_pad_m, py.max() + bbox_pad_m
        keep = ~((np.maximum(ax, bx) < lo_x) | (np.minimum(ax, bx) > hi_x) |
                 (np.maximum(ay, by) < lo_y) | (np.minimum(ay, by) > hi_y))
        ids = np.where(keep)[0]
        length = np.hypot(bx - ax, by - ay)[ids]
        n_samp = np.maximum(1, np.ceil(length / SAMPLE_STEP_M).astype(int)) + 1
        arc_of = np.repeat(ids, n_samp)
        frac = np.concatenate([np.linspace(0, 1, k) for k in n_samp])
        sx = ax[arc_of] + frac * (bx[arc_of] - ax[arc_of])
        sy = ay[arc_of] + frac * (by[arc_of] - ay[arc_of])
        self.tree = cKDTree(np.c_[sx, sy])
        self.sample_arc = arc_of

    def candidates(self, px, py, radius_m: float, k: int) -> pd.DataFrame:
        """Return long table: point, arc, dist_m, t (projection fraction), rank."""
        hits = self.tree.query_ball_point(np.c_[px, py], r=radius_m + SAMPLE_STEP_M / 2)
        pt = np.repeat(np.arange(len(px)), [len(h) for h in hits])
        arc = self.sample_arc[np.concatenate(hits).astype(int)] if len(pt) else np.empty(0, int)
        df = pd.DataFrame({"point": pt, "arc": arc}).drop_duplicates()
        p, a = df["point"].to_numpy(), df["arc"].to_numpy()
        d, t = point_segment_distance(px[p], py[p], self.nx[self.u[a]], self.ny[self.u[a]],
                                      self.nx[self.v[a]], self.ny[self.v[a]])
        df["dist_m"], df["t"] = d, t
        df = df[df["dist_m"] <= radius_m].sort_values(["point", "dist_m", "arc"])
        df["rank"] = df.groupby("point").cumcount()
        # two-way streets give tied distances, so cut on distinct distance rank, keeping both directions
        df["dist_rank"] = df.groupby("point")["dist_m"].rank(method="dense").astype(int) - 1
        return df[df["dist_rank"] < k].reset_index(drop=True)


def generate(trip: Trip, radius_m: float, k: int, proj: LocalProjection | None = None) -> pd.DataFrame:
    proj = proj or LocalProjection.for_track(trip.track)
    px, py = proj.to_xy(trip.track["lon"].to_numpy(), trip.track["lat"].to_numpy())
    return ArcIndex(trip, proj).candidates(px, py, radius_m, k)
