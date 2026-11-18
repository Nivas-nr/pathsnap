"""Planar geometry helpers (inputs in projected metres)."""
import numpy as np


def point_segment_distance(px, py, ax, ay, bx, by):
    """Distance from points P to segments AB, plus projection fraction t in [0, 1].

    All inputs broadcast. t = 0 means the closest point is A, t = 1 means B.
    """
    dx, dy = bx - ax, by - ay
    len2 = dx * dx + dy * dy
    t = np.where(len2 > 0, ((px - ax) * dx + (py - ay) * dy) / np.where(len2 > 0, len2, 1), 0.0)
    t = np.clip(t, 0.0, 1.0)
    cx, cy = ax + t * dx, ay + t * dy
    return np.hypot(px - cx, py - cy), t


def bearing_deg(dx, dy):
    """Compass bearing (0 = north, 90 = east) of a planar vector."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff_deg(a, b):
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)
