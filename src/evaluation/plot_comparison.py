"""Plots for the model comparison (all from results/models/*.csv|json).

Run:  python -m src.evaluation.plot_comparison
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.data.loader import ROOT
from src.features.build_features import FEATURES

RES = ROOT / "results" / "models"
FIG = ROOT / "reports" / "figures" / "models"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    comp = pd.read_csv(RES / "comparison.csv")
    folds = pd.read_csv(RES / "cv_folds.csv")
    ml = comp[comp.model != "nearest_arc"].reset_index(drop=True)
    base = comp[comp.model == "nearest_arc"].iloc[0]

    # 1. CV macro F1 per fold (box) - shows stability across folds
    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = ml.sort_values("cv_macro_f1")["model"].tolist()
    ax.boxplot([folds[folds.model == m]["macro_f1"] for m in order], tick_labels=order)
    ax.scatter(range(1, len(order) + 1), ml.set_index("model").loc[order, "cv_macro_f1"], color="red", zorder=3,
               label="mean")
    ax.set(title="5-fold GroupKFold CV Macro F1 (train trips)", ylabel="Macro F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "cv_macro_f1.png", dpi=120)

    # 2. student-test top-1 accuracy and macro F1
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(ml))
    ax.bar(x - 0.2, ml["test_top1_acc"], 0.4, label="Top-1 accuracy")
    ax.bar(x + 0.2, ml["test_macro_f1"], 0.4, label="Macro F1")
    ax.axhline(base["test_top1_acc"], color="grey", ls="--", label="nearest-arc baseline top-1")
    ax.set_xticks(x, ml["model"])
    ax.set(ylim=(0.4, 1.0), title="Student-test performance", ylabel="score")
    for i, (a, f) in enumerate(zip(ml["test_top1_acc"], ml["test_macro_f1"])):
        ax.text(i - 0.2, a + 0.005, f"{a:.3f}", ha="center", fontsize=7)
        ax.text(i + 0.2, f + 0.005, f"{f:.3f}", ha="center", fontsize=7)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "student_test_scores.png", dpi=120)

    # 3. log loss (CV and test)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.2, ml["cv_log_loss"], 0.4, label="CV")
    ax.bar(x + 0.2, ml["test_log_loss"], 0.4, label="student-test")
    ax.set_xticks(x, ml["model"])
    ax.set(yscale="log", title="Log loss (lower is better)", ylabel="log loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "log_loss.png", dpi=120)

    # 4. accuracy vs inference cost
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(ml["infer_ms_per_1k_points"], ml["test_top1_acc"])
    for _, r in ml.iterrows():
        ax.annotate(r["model"], (r["infer_ms_per_1k_points"], r["test_top1_acc"]))
    ax.set(xscale="log", xlabel="inference ms per 1000 GPS points", ylabel="student-test top-1",
           title="Accuracy vs inference cost")
    fig.tight_layout()
    fig.savefig(FIG / "accuracy_vs_speed.png", dpi=120)

    # 5. MLP training history (only model with epoch-wise history)
    for f in RES.glob("history_*.json"):
        h = json.loads(f.read_text())
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(h["train_loss"], label="train loss")
        ax1.set(xlabel="epoch", ylabel="log loss", title=f"{f.stem.split('_', 1)[1]} training history")
        if h["val_score"]:
            ax2 = ax1.twinx()
            ax2.plot(h["val_score"], color="orange", label="validation accuracy")
            ax2.set_ylabel("validation accuracy (held-out 10%)")
            ax2.legend(loc="center right")
        ax1.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(FIG / f"{f.stem}.png", dpi=120)

    # 6. confusion matrices on student-test (row level, threshold 0.5)
    test = pd.read_parquet(ROOT / "data" / "processed" / "features_student_test.parquet")
    lab = (test.label >= 0).to_numpy()
    fig, axes = plt.subplots(1, len(ml), figsize=(3.2 * len(ml), 3.2))
    for ax, m in zip(axes, ml["model"]):
        prob = np.load(RES / f"student_test_prob_{m}.npy")
        cm = confusion_matrix(test["is_true"].to_numpy()[lab], (prob[lab] >= 0.5).astype(int), normalize="true")
        ConfusionMatrixDisplay(cm, display_labels=["wrong arc", "true arc"]).plot(ax=ax, colorbar=False,
                                                                                  values_format=".2f")
        ax.set_title(m)
    fig.tight_layout()
    fig.savefig(FIG / "confusion_matrices.png", dpi=110)

    # 7. LightGBM feature importance (gain)
    lgbm = joblib.load(ROOT / "artifacts" / "models" / "lgbm.joblib")
    imp = pd.Series(lgbm.booster_.feature_importance("gain"), index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(7, 6))
    (imp / imp.sum()).plot.barh(ax=ax)
    ax.set(title="LightGBM feature importance (gain share)")
    fig.tight_layout()
    fig.savefig(FIG / "lgbm_feature_importance.png", dpi=120)
    print("figures in", FIG)


if __name__ == "__main__":
    main()
