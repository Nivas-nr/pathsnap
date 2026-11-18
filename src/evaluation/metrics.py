"""Evaluation metrics for candidate classification and route quality."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score

from src.data.loader import list_trips, load_trip


def row_metrics(y_true, prob, threshold=0.5):
    """Binary candidate classification: is this candidate the true arc?"""
    pred = (prob >= threshold).astype(int)
    return dict(
        macro_f1=f1_score(y_true, pred, average="macro"),
        precision=precision_score(y_true, pred, zero_division=0),
        recall=recall_score(y_true, pred, zero_division=0),
        log_loss=log_loss(y_true, np.clip(prob, 1e-7, 1 - 1e-7), labels=[0, 1]),
    )


def top1_choice(df: pd.DataFrame, score_col="prob") -> pd.DataFrame:
    """Highest-scoring candidate per (trip, point). Ties broken by distance."""
    order = df.sort_values(["trip", "point", score_col, "dist_m"], ascending=[True, True, False, True])
    return order.drop_duplicates(["trip", "point"])


def top1_accuracy(df: pd.DataFrame, points_with_label: pd.DataFrame, score_col="prob") -> float:
    """% of labelled points whose top candidate is the reference arc.

    `points_with_label` lists every labelled point (trip, point), including those
    whose true arc fell outside the candidate set - those count as errors.
    """
    best = top1_choice(df, score_col)[["trip", "point", "is_true"]]
    m = points_with_label.merge(best, on=["trip", "point"], how="left")
    return float(m["is_true"].fillna(0).mean())


class ArcTables:
    """Cache of arc endpoints per trip (for route-continuity checks)."""

    def __init__(self, partition):
        self.uv = {}
        for tid in list_trips(partition):
            arcs = load_trip(tid, partition).arcs
            self.uv[tid] = (arcs["u"].to_numpy(), arcs["v"].to_numpy())


def route_continuity(arc_seq: np.ndarray, u: np.ndarray, v: np.ndarray) -> float:
    """% of transitions between consecutive distinct arcs that are physically connected."""
    seq = arc_seq[np.r_[True, arc_seq[1:] != arc_seq[:-1]]]
    if len(seq) < 2:
        return 1.0
    return float(np.mean(v[seq[:-1]] == u[seq[1:]]))


def mean_route_continuity(choice: pd.DataFrame, arc_tables: ArcTables) -> float:
    vals = []
    for tid, g in choice.sort_values(["trip", "point"]).groupby("trip"):
        u, v = arc_tables.uv[tid]
        vals.append(route_continuity(g["arc"].to_numpy(), u, v))
    return float(np.mean(vals))
