"""audio_engine/plots.py

Graphiques du benchmark (pour le rapport et pour MLflow).

Appelé automatiquement à la fin de benchmark.py, ou à la demande :
    python -m audio_engine.plot --out-dir audio_engine

Fichiers lus dans --out-dir :
    benchmark_results.csv   (obligatoire)  -> barres de métriques, efficacité
    predictions.npz         (optionnel)    -> courbes ROC / PR, matrices de confusion
    histories.json          (optionnel)    -> courbes d'apprentissage
    onnx_meta.json          (optionnel)    -> nom du modèle retenu (marqué d'une étoile)
Les figures sont écrites dans <out-dir>/plots/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (average_precision_score, confusion_matrix,  # noqa: E402
                             precision_recall_curve, roc_auc_score, roc_curve)

COLORS = {"decision_tree": "#7f7f7f", "svm_rbf": "#1f77b4", "cnn_custom": "#d62728",
          "resnet_transfer": "#2ca02c", "transformer_cnn": "#9467bd"}
LABELS = {"decision_tree": "Decision Tree", "svm_rbf": "SVM (RBF)", "cnn_custom": "CNN custom",
          "resnet_transfer": "ResNet (TL)", "transformer_cnn": "Transformer-CNN"}


def _lab(m: str, best: str | None) -> str:
    return LABELS.get(m, m) + (" ★" if m == best else "")


def _col(m: str) -> str:
    return COLORS.get(m, "#333333")


def _save(fig, out: Path, name: str) -> Path:
    path = out / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------------------------------------- à partir du CSV
def plot_metrics(df: pd.DataFrame, out: Path, best: str | None) -> Path:
    """Recall / spécificité / F1 / AUC sur le jeu de TEST pour chaque algorithme."""
    metrics = [("recall", "Recall (classe anormale)"), ("specificity", "Spécificité"),
               ("f1", "F1"), ("auc", "AUC")]
    x, w = np.arange(len(df)), 0.2
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for i, (k, lab) in enumerate(metrics):
        vals = df[f"test_{k}"].fillna(0).to_numpy()
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=lab)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=1)
    ax.set_xticks(x, [_lab(m, best) for m in df["model"]], rotation=15)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score (jeu de test)")
    ax.set_title("Comparaison des 5 algorithmes — jeu de test")
    ax.legend(ncol=4, loc="upper center", fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out, "1_metrics_comparison.png")


def plot_recall_val_test(df: pd.DataFrame, out: Path, best: str | None) -> Path:
    """Recall validation vs test : un grand écart = le modèle généralise mal."""
    x, w = np.arange(len(df)), 0.35
    fig, ax = plt.subplots(figsize=(8, 4.2))
    b1 = ax.bar(x - w / 2, df["val_recall"].fillna(0), w, label="Validation (sert à choisir)")
    b2 = ax.bar(x + w / 2, df["test_recall"].fillna(0), w, label="Test (estimation honnête)")
    ax.bar_label(b1, fmt="%.2f", fontsize=8, padding=1)
    ax.bar_label(b2, fmt="%.2f", fontsize=8, padding=1)
    ax.set_xticks(x, [_lab(m, best) for m in df["model"]], rotation=15)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Recall (classe anormale)")
    ax.set_title("Recall : validation vs test")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out, "2_recall_val_vs_test.png")


def plot_efficiency(df: pd.DataFrame, out: Path, best: str | None) -> Path:
    """Coût : temps d'entraînement, latence d'inférence, taille du modèle."""
    panels = [("train_time_s", "Temps d'entraînement (s)", False),
              ("latency_ms_per_segment", "Latence (ms / segment)", False),
              ("n_params", "Paramètres (échelle log)", True)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    names = [_lab(m, best) for m in df["model"]]
    for ax, (col, title, log) in zip(axes, panels):
        if col not in df:
            ax.set_visible(False)
            continue
        vals = df[col].astype(float).to_numpy()
        ax.bar(range(len(df)), vals, color=[_col(m) for m in df["model"]])
        ax.set_xticks(range(len(df)), names, rotation=40, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        if log:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, out, "7_efficiency.png")


# ------------------------------------------------------- à partir des prédictions
def plot_roc(y: np.ndarray, proba: dict, out: Path, best: str | None) -> Path:
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    for m, p in proba.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=_col(m), lw=2 if m == best else 1.4,
                label=f"{_lab(m, best)} (AUC={roc_auc_score(y, p):.3f})")
        pred = p >= 0.5       # point de fonctionnement au seuil 0.5
        ax.scatter([((pred == 1) & (y == 0)).sum() / max((y == 0).sum(), 1)],
                   [((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)],
                   color=_col(m), s=28, zorder=3)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("Taux de faux positifs (1 − spécificité)")
    ax.set_ylabel("Taux de vrais positifs (recall)")
    ax.set_title("Courbes ROC — test (points = seuil 0.5)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    return _save(fig, out, "3_roc_curves.png")


def plot_pr(y: np.ndarray, proba: dict, out: Path, best: str | None) -> Path:
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    for m, p in proba.items():
        pr, rc, _ = precision_recall_curve(y, p)
        ax.plot(rc, pr, color=_col(m), lw=2 if m == best else 1.4,
                label=f"{_lab(m, best)} (AP={average_precision_score(y, p):.3f})")
    ax.axhline(y.mean(), color="k", ls="--", lw=0.8, label=f"Hasard ({y.mean():.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Précision")
    ax.set_title("Courbes précision-recall — test")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    return _save(fig, out, "4_precision_recall.png")


def plot_confusions(y: np.ndarray, proba: dict, out: Path, best: str | None) -> Path:
    n = len(proba)
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 3.5), squeeze=False)
    for ax, (m, p) in zip(axes[0], proba.items()):
        cm = confusion_matrix(y, (p >= 0.5).astype(int), labels=[0, 1])
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v}\n({norm[i, j]:.0%})", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
        ax.set_xticks([0, 1], ["normal", "anormal"], fontsize=8)
        ax.set_yticks([0, 1], ["normal", "anormal"], fontsize=8)
        ax.set_xlabel("Prédit", fontsize=8)
        ax.set_title(_lab(m, best), fontsize=9)
    axes[0][0].set_ylabel("Réel", fontsize=8)
    fig.suptitle("Matrices de confusion — test (% par classe réelle)", y=1.03)
    fig.tight_layout()
    return _save(fig, out, "5_confusion_matrices.png")


# ------------------------------------------------------- à partir des historiques
def plot_training(histories: dict, out: Path, best: str | None) -> list[Path]:
    hist = {m: pd.DataFrame(h) for m, h in histories.items() if h}
    if not hist:
        return []
    paths = []
    for m, h in hist.items():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        c, lab = _col(m), _lab(m, best)
        axes[0].plot(h["epoch"], h["train_loss"], color=c, label=f"{lab} (train)")
        axes[0].plot(h["epoch"], h["val_loss"], color="#ff7f0e", ls="--",
                     label="Validation")
        axes[0].set_title("Perte : entraînement vs validation", fontsize=10)
        axes[0].set_ylabel("Cross-entropy")
        axes[1].plot(h["epoch"], h["val_recall"], color="#2ca02c", label="Recall validation")
        axes[1].plot(h["epoch"], h["val_auc"], color="#9467bd", label="AUC validation")
        axes[1].set_title("Métriques sur la validation", fontsize=10)
        axes[1].set_ylim(0, 1.05)
        for ax in axes:
            ax.set_xlabel("Époque")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle(f"Courbes d'apprentissage — {lab}")
        fig.text(
            0.5, 0.01,
            "Surapprentissage possible : perte train en baisse et perte val qui remonte/s'écarte. "
            "Sous-apprentissage possible : pertes encore élevées et métriques faibles.",
            ha="center", fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.08, 1, 0.94))
        paths.append(_save(fig, out, f"6_training_curves_{m}.png"))
    return paths


# ---------------------------------------------------------------------- tout
def make_all(out_dir, df: pd.DataFrame, preds: dict | None = None,
             histories: dict | None = None, best: str | None = None) -> list[Path]:
    """preds = {"y": array test, "proba": {modèle: array}}. Retourne les fichiers créés."""
    out = Path(out_dir) / "plots"
    out.mkdir(parents=True, exist_ok=True)
    paths = [plot_metrics(df, out, best), plot_recall_val_test(df, out, best), plot_efficiency(df, out, best)]
    if preds and preds.get("proba"):
        y, proba = np.asarray(preds["y"]), preds["proba"]
        if len(np.unique(y)) == 2:
            paths += [plot_roc(y, proba, out, best), plot_pr(y, proba, out, best),
                      plot_confusions(y, proba, out, best)]
    if histories:
        paths.extend(plot_training(histories, out, best))
    return sorted(paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=Path(__file__).resolve().parent)
    a, _ = ap.parse_known_args()
    d = Path(a.out_dir)
    df = pd.read_csv(d / "benchmark_results.csv")

    preds = None
    if (d / "predictions.npz").exists():
        z = np.load(d / "predictions.npz")
        preds = {"y": z["y_test"], "proba": {k[2:]: z[k] for k in z.files if k.startswith("p_")}}
    histories = json.loads((d / "histories.json").read_text()) if (d / "histories.json").exists() else None
    best = None
    if (d / "onnx_meta.json").exists():
        best = json.loads((d / "onnx_meta.json").read_text()).get("algo")

    paths = make_all(d, df, preds, histories, best)
    if preds is None:
        print("predictions.npz absent : ROC, PR et matrices de confusion ignorées "
              "(relance benchmark.py pour les générer).")
    if histories is None:
        print("histories.json absent : courbes d'apprentissage ignorées "
              "(relance benchmark.py pour entraîner les modèles et les générer).")
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()