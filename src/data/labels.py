"""Derive a per-GPS-point label (arc_id) from the trip-level reference route.

The .route file lists the arcs driven, in order - it does NOT say which GPS
point lies on which arc. We recover that with a monotonic alignment:

    point i is assigned route position j_i, with j_0 <= j_1 <= ... <= j_{N-1}

minimising the total point-to-arc distance. This respects driving order, so a
route that revisits a street or runs beside a parallel road still gets correct
labels (a "nearest route arc" rule would not).

Dynamic programme (N points x M route positions):
    D[i, j] = dist(i, route[j]) + min_{k <= j} D[i-1, k]
The inner min is a running prefix-minimum, so each row costs O(M).
Positions may be skipped (short arcs that got no GPS fix), which is allowed.

Points whose assigned arc is farther than MAX_LABEL_DIST_M are labelled -1
(parked, off-network or bad fixes) and excluded from training/evaluation.
"""
import numpy as np

from src.data.loader import LocalProjection, Trip
from src.features.geometry import point_segment_distance

MAX_LABEL_DIST_M = 50.0


def align_route(trip: Trip, proj: LocalProjection | None = None, max_dist: float = MAX_LABEL_DIST_M):
    """Return (labels[N] arc_id or -1, route_pos[N], dist[N] metres)."""
    proj = proj or LocalProjection.for_track(trip.track)
    route = trip.route
    nx, ny = proj.to_xy(trip.nodes["lon"].to_numpy(), trip.nodes["lat"].to_numpy())
    u, v = trip.arcs["u"].to_numpy()[route], trip.arcs["v"].to_numpy()[route]
    ax, ay, bx, by = nx[u], ny[u], nx[v], ny[v]
    px, py = proj.to_xy(trip.track["lon"].to_numpy(), trip.track["lat"].to_numpy())

    n, m = len(px), len(route)
    back = np.empty((n, m), dtype=np.int16 if m < 32767 else np.int32)
    prev = None
    for i in range(n):
        cost, _ = point_segment_distance(px[i], py[i], ax, ay, bx, by)
        if prev is None:
            cur = cost
            back[i] = np.arange(m)
        else:
            # prefix minimum of prev and where it came from
            pm = np.minimum.accumulate(prev)
            is_new = np.empty(m, bool)
            is_new[0] = True
            is_new[1:] = prev[1:] < pm[:-1]
            arg = np.maximum.accumulate(np.where(is_new, np.arange(m), 0))
            cur = cost + pm
            back[i] = arg
        prev = cur

    pos = np.empty(n, dtype=np.int64)
    pos[-1] = int(np.argmin(prev))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]

    dist, _ = point_segment_distance(px, py, ax[pos], ay[pos], bx[pos], by[pos])
    labels = np.where(dist <= max_dist, route[pos], -1)
    return labels, pos, dist
