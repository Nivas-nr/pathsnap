"""Train and compare candidate classifiers from several algorithm families.

Protocol
  * Training rows: candidates of labelled points (label >= 0) from the 80 train trips.
  * 5-fold GroupKFold by trip -> CV metrics (no trip appears in train and validation).
  * Refit on all train trips -> evaluate on the 10 student_test trips.
  * Scaling is fitted inside each pipeline, i.e. on training folds only.

Run:  python -m src.models.train_compare            (all models)
      python -m src.models.train_compare lgbm lr    (subset)
"""
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.loader import ROOT
from src.evaluation.metrics import ArcTables, mean_route_continuity, row_metrics, top1_accuracy, top1_choice
from src.features.build_features import FEATURES

SEED = 42
N_FOLDS = 5
OUT = ROOT / "results" / "models"
ART = ROOT / "artifacts" / "models"

# name -> (family, factory, max training rows or None, config summary)
MODELS = {
    "lr": ("Linear", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)),
           None, "StandardScaler + L2 LogisticRegression C=1"),
    "gnb": ("Probabilistic", lambda: make_pipeline(StandardScaler(), GaussianNB()),
            None, "GaussianNB, var_smoothing=1e-9"),
    "knn": ("Instance-based", lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(
        n_neighbors=50, weights="distance", n_jobs=-1)), 100_000, "k=50 distance-weighted, 100k-row subsample"),
    "rf": ("Bagging ensemble", lambda: RandomForestClassifier(
        n_estimators=150, max_depth=20, min_samples_leaf=5, max_samples=0.3, n_jobs=-1, random_state=SEED),
           None, "150 trees, depth<=20, leaf>=5, 30% bootstrap"),
    "lgbm": ("Boosting", lambda: LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, random_state=SEED, verbose=-1), None, "400 trees, lr=0.05, 63 leaves"),
    "mlp": ("Neural network", lambda: make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=1024, learning_rate_init=1e-3, max_iter=40,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=5, random_state=SEED)),
            None, "MLP 64-32 ReLU, Adam, early stopping"),
}


def load(part):
    df = pd.read_parquet(ROOT / "data" / "processed" / f"features_{part}.parquet")
    return df


def labelled_points(part):
    """All labelled points of a partition, including those with no candidates."""
    rows = []
    for f in sorted((ROOT / "data" / "processed" / "labels" / part).glob("*.npz")):
        lab = np.load(f)["labels"]
        pts = np.where(lab >= 0)[0]
        rows.append(pd.DataFrame({"trip": f.stem, "point": pts}))
    return pd.concat(rows, ignore_index=True)


def subsample(X, y, cap, rng):
    if cap is None or len(X) <= cap:
        return X, y
    i = rng.choice(len(X), cap, replace=False)
    return X[i], y[i]


def evaluate(df_eval, prob, lab_points, arc_tables):
    df_eval = df_eval.assign(prob=prob)
    lab = df_eval[df_eval.label >= 0]
    out = row_metrics(lab["is_true"].to_numpy(), lab["prob"].to_numpy())
    out["top1_acc"] = top1_accuracy(df_eval, lab_points)
    out["route_continuity"] = mean_route_continuity(top1_choice(df_eval), arc_tables)
    return out


def run_model(name, train, test, train_pts, test_pts, arcs_train, arcs_test):
    family, factory, cap, config = MODELS[name]
    rng = np.random.default_rng(SEED)
    tr = train[train.label >= 0].reset_index(drop=True)
    X, y, groups = tr[FEATURES].to_numpy(np.float32), tr["is_true"].to_numpy(), tr["trip"].to_numpy()

    folds = []
    for k, (fi, vi) in enumerate(GroupKFold(n_splits=N_FOLDS).split(X, y, groups)):
        Xf, yf = subsample(X[fi], y[fi], cap, rng)
        model = factory()
        t0 = time.time()
        model.fit(Xf, yf)
        fit_s = time.time() - t0
        val_trips = np.unique(groups[vi])
        val_df = train[train.trip.isin(val_trips)]
        prob = model.predict_proba(val_df[FEATURES].to_numpy(np.float32))[:, 1]
        m = evaluate(val_df, prob, train_pts[train_pts.trip.isin(val_trips)],
                     arcs_train)
        m.update(fold=k, fit_s=fit_s)
        folds.append(m)
        print(f"  [{name}] fold {k}: macroF1={m['macro_f1']:.4f} top1={m['top1_acc']:.4f} "
              f"logloss={m['log_loss']:.4f} fit={fit_s:.0f}s", flush=True)
    folds = pd.DataFrame(folds)

    # refit on all training trips, evaluate on student_test
    Xa, ya = subsample(X, y, cap, rng)
    model = factory()
    t0 = time.time()
    model.fit(Xa, ya)
    fit_s = time.time() - t0
    Xt = test[FEATURES].to_numpy(np.float32)
    t0 = time.time()
    prob = model.predict_proba(Xt)[:, 1]
    infer_s = time.time() - t0
    st = evaluate(test, prob, test_pts, arcs_test)

    ART.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ART / f"{name}.joblib")
    np.save(OUT / f"student_test_prob_{name}.npy", prob)
    save_history(name, model)

    summary = dict(
        model=name, family=family, config=config,
        cv_macro_f1=folds.macro_f1.mean(), cv_macro_f1_std=folds.macro_f1.std(),
        cv_top1_acc=folds.top1_acc.mean(), cv_log_loss=folds.log_loss.mean(),
        cv_route_continuity=folds.route_continuity.mean(),
        test_macro_f1=st["macro_f1"], test_precision=st["precision"], test_recall=st["recall"],
        test_top1_acc=st["top1_acc"], test_log_loss=st["log_loss"],
        test_route_continuity=st["route_continuity"],
        train_time_s=fit_s,
        infer_ms_per_1k_points=infer_s * 1e6 / test[["trip", "point"]].drop_duplicates().shape[0],
    )
    return summary, folds.assign(model=name)


def save_history(name, model):
    """Epoch-wise loss only exists for MLP (sklearn records it)."""
    est = model[-1] if hasattr(model, "steps") else model
    if hasattr(est, "loss_curve_"):
        hist = {"train_loss": list(map(float, est.loss_curve_)),
                "val_score": list(map(float, getattr(est, "validation_scores_", []) or []))}
        (OUT / f"history_{name}.json").write_text(json.dumps(hist))


def baseline_nearest(test, test_pts, arcs_test, train, train_pts, arcs_train):
    """Non-ML reference: score = -distance (pick the nearest arc)."""
    res = {}
    for split, df, pts, arcs in [("test", test, test_pts, arcs_test), ("cv", train, train_pts, arcs_train)]:
        d = df.assign(prob=-df["dist_m"])
        res[f"{split}_top1_acc"] = top1_accuracy(d, pts)
        res[f"{split}_route_continuity"] = mean_route_continuity(top1_choice(d), arcs)
    return dict(model="nearest_arc", family="Heuristic baseline", config="pick minimum distance", **res)


def main(names):
    OUT.mkdir(parents=True, exist_ok=True)
    train, test = load("train"), load("student_test")
    train_pts, test_pts = labelled_points("train"), labelled_points("student_test")
    arcs_train, arcs_test = ArcTables("train"), ArcTables("student_test")

    summaries, all_folds = [baseline_nearest(test, test_pts, arcs_test, train, train_pts, arcs_train)], []
    print("baseline", summaries[0])
    for name in names:
        print(f"== {name}", flush=True)
        s, f = run_model(name, train, test, train_pts, test_pts, arcs_train, arcs_test)
        summaries.append(s)
        all_folds.append(f)
        print(pd.Series(s).to_string(), flush=True)
        # write incrementally so partial runs keep their results
        pd.DataFrame(summaries).to_csv(OUT / "comparison_partial.csv", index=False)

    pd.DataFrame(summaries).to_csv(OUT / "comparison.csv", index=False)
    pd.concat(all_folds).to_csv(OUT / "cv_folds.csv", index=False)
    print(pd.DataFrame(summaries).round(4).to_string())


if __name__ == "__main__":
    main(sys.argv[1:] or list(MODELS))
