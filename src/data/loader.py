"""Load PathSnap trips from the raw dataset.

Each trip folder <id>/ holds four headerless, whitespace-separated files:
  <id>.track  lon lat t     GPS observations (~1 Hz)
  <id>.nodes  lon lat       road-network nodes, row index = node_id
  <id>.arcs   from to       road segments, row index = arc_id
  <id>.route  arc_id        reference route: ordered arc_ids driven
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "Student folder" / "Student folder"
EARTH_RADIUS_M = 6_371_000.0


@dataclass
class Trip:
    trip_id: str
    track: pd.DataFrame   # lon, lat, t
    nodes: pd.DataFrame   # lon, lat   (index = node_id)
    arcs: pd.DataFrame    # u, v       (index = arc_id)
    route: np.ndarray | None  # arc_ids in driving order


def list_trips(partition: str, data_dir: Path = DEFAULT_DATA_DIR) -> list[str]:
    return sorted(p.name for p in (Path(data_dir) / partition).iterdir() if p.is_dir())


def _read(path: Path, names: list[str], dtype) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype, engine="c")


def load_trip(trip_id: str, partition: str, data_dir: Path = DEFAULT_DATA_DIR) -> Trip:
    d = Path(data_dir) / partition / trip_id
    track = _read(d / f"{trip_id}.track", ["lon", "lat", "t"], float)
    nodes = _read(d / f"{trip_id}.nodes", ["lon", "lat"], float)
    arcs = _read(d / f"{trip_id}.arcs", ["u", "v"], np.int64)
    route_path = d / f"{trip_id}.route"
    route = _read(route_path, ["arc"], np.int64)["arc"].to_numpy() if route_path.exists() else None
    return Trip(trip_id, track, nodes, arcs, route)


class LocalProjection:
    """Equirectangular projection to metres around a reference point.

    Lon/lat are angles, so we can't take Euclidean distance in degrees: at 49.9 deg N
    one degree of longitude is ~72 km but one degree of latitude is ~111 km.
    Each trip covers only a few km, so a flat tangent plane centred on the trip
    has error well under 0.1% - far below GPS noise (several metres).
    """

    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = lon0, lat0
        self.kx = np.deg2rad(1.0) * EARTH_RADIUS_M * np.cos(np.deg2rad(lat0))
        self.ky = np.deg2rad(1.0) * EARTH_RADIUS_M

    @classmethod
    def for_track(cls, track: pd.DataFrame) -> "LocalProjection":
        return cls(track["lon"].mean(), track["lat"].mean())

    def to_xy(self, lon, lat):
        return (np.asarray(lon) - self.lon0) * self.kx, (np.asarray(lat) - self.lat0) * self.ky


def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres (used to sanity-check the projection)."""
    lon1, lat1, lon2, lat2 = map(np.deg2rad, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))
