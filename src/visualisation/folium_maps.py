"""Interactive Folium maps: GPS points, nearby road network, reference and predicted routes.

Run:  python -m src.visualisation.folium_maps train 00000001
Writes reports/maps/<partition>_<trip>.html
"""
import sys

import folium
import numpy as np
from scipy.spatial import cKDTree

from src.data.loader import ROOT, LocalProjection, Trip, load_trip

OUT = ROOT / "reports" / "maps"


def _arc_lines(trip: Trip, arc_ids):
    lon, lat = trip.nodes["lon"].to_numpy(), trip.nodes["lat"].to_numpy()
    u, v = trip.arcs["u"].to_numpy()[arc_ids], trip.arcs["v"].to_numpy()[arc_ids]
    return [[(lat[a], lon[a]), (lat[b], lon[b])] for a, b in zip(u, v)]


def _route_polylines(trip: Trip, route):
    """Join consecutive connected arcs into polylines (fewer, smoother map objects)."""
    lon, lat = trip.nodes["lon"].to_numpy(), trip.nodes["lat"].to_numpy()
    u, v = trip.arcs["u"].to_numpy(), trip.arcs["v"].to_numpy()
    lines, cur = [], []
    for a in route:
        if cur and u[a] != cur_end:
            lines.append(cur)
            cur = []
        if not cur:
            cur = [(lat[u[a]], lon[u[a]])]
        cur.append((lat[v[a]], lon[v[a]]))
        cur_end = v[a]
    if cur:
        lines.append(cur)
    return lines


def nearby_arcs(trip: Trip, radius_m=150.0):
    """Arcs with an endpoint within radius_m of any GPS point (keeps the HTML small)."""
    proj = LocalProjection.for_track(trip.track)
    nx, ny = proj.to_xy(trip.nodes["lon"].to_numpy(), trip.nodes["lat"].to_numpy())
    px, py = proj.to_xy(trip.track["lon"].to_numpy(), trip.track["lat"].to_numpy())
    d, _ = cKDTree(np.c_[px, py]).query(np.c_[nx, ny], distance_upper_bound=radius_m)
    near = np.isfinite(d)
    u, v = trip.arcs["u"].to_numpy(), trip.arcs["v"].to_numpy()
    return np.where(near[u] | near[v])[0]


def trip_map(trip: Trip, predicted_route=None, point_labels=None, max_points=3000, title=None):
    tr = trip.track
    m = folium.Map(location=[tr["lat"].mean(), tr["lon"].mean()], zoom_start=14, tiles="cartodbpositron")

    net = folium.FeatureGroup(name="Road network (nearby)")
    for line in _arc_lines(trip, nearby_arcs(trip)):
        folium.PolyLine(line, color="#9e9e9e", weight=1, opacity=0.6).add_to(net)
    net.add_to(m)

    if trip.route is not None:
        ref = folium.FeatureGroup(name="Reference route")
        for line in _route_polylines(trip, trip.route):
            folium.PolyLine(line, color="#2e7d32", weight=6, opacity=0.7, tooltip="reference").add_to(ref)
        ref.add_to(m)

    if predicted_route is not None:
        pred = folium.FeatureGroup(name="Predicted route")
        for line in _route_polylines(trip, predicted_route):
            folium.PolyLine(line, color="#d32f2f", weight=3, opacity=0.9, dash_array="6 4",
                            tooltip="predicted").add_to(pred)
        pred.add_to(m)

    gps = folium.FeatureGroup(name="GPS observations")
    step = max(1, len(tr) // max_points)
    for i in range(0, len(tr), step):
        bad = point_labels is not None and point_labels[i] < 0
        folium.CircleMarker((tr["lat"].iat[i], tr["lon"].iat[i]), radius=2, weight=0, fill=True,
                            fill_color="#ff9800" if bad else "#1565c0", fill_opacity=0.8,
                            tooltip=f"#{i} t={tr['t'].iat[i]:.0f}s" + (" (unlabelled)" if bad else "")).add_to(gps)
    gps.add_to(m)

    folium.Marker((tr["lat"].iat[0], tr["lon"].iat[0]), tooltip="start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker((tr["lat"].iat[-1], tr["lon"].iat[-1]), tooltip="end", icon=folium.Icon(color="red")).add_to(m)

    legend = f"""<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;
        padding:8px 12px;border:1px solid #999;border-radius:6px;font:13px sans-serif">
        <b>{title or 'Trip ' + trip.trip_id}</b><br>
        <span style="color:#1565c0">&#9679;</span> GPS point<br>
        <span style="color:#ff9800">&#9679;</span> GPS point, unlabelled (&gt;50 m from route)<br>
        <span style="color:#9e9e9e">&#9473;</span> Road network<br>
        <span style="color:#2e7d32">&#9473;</span> Reference route<br>
        {'<span style="color:#d32f2f">&#9476;</span> Predicted route<br>' if predicted_route is not None else ''}
        </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[tr["lat"].min(), tr["lon"].min()], [tr["lat"].max(), tr["lon"].max()]])
    return m


def main(partition="train", trip_id="00000001"):
    trip = load_trip(trip_id, partition)
    lab_file = ROOT / "data" / "processed" / "labels" / partition / f"{trip_id}.npz"
    labels = np.load(lab_file)["labels"] if lab_file.exists() else None
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{partition}_{trip_id}.html"
    trip_map(trip, point_labels=labels).save(out)
    print("saved", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
