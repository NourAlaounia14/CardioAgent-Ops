"""audio_engine/benchmark.py

Benchmark des 5 algorithmes sur les données produites par preprocessing.py,
suivi complet sous MLflow (run parent + 5 runs imbriqués nested=True),
sélection du meilleur modèle sur le RECALL de la classe pathologique (validation),
export ONNX (cardio_model.onnx) et enregistrement dans le Model Registry.

Algorithmes :
    1. decision_tree     - Decision Tree      (features 1D : MFCC + ZCR)
    2. svm_rbf           - SVM noyau RBF      (features 1D)
    3. cnn_custom        - CNN 2D custom      (spectrogramme Mel)
    4. resnet_transfer   - ResNet18/50 pré-entraîné ImageNet (spectrogramme Mel)
    5. transformer_cnn   - Hybride CNN + Transformer encoder (spectrogramme Mel)

Usage :
    python -m audio_engine.benchmark --data-dir data/processed
    python -m audio_engine.benchmark --epochs 40 --models cnn_custom transformer_cnn
    MLFLOW_TRACKING_URI=http://localhost:5000 python -m audio_engine.benchmark

Sorties (dans --out-dir, défaut audio_engine/models) :
    cardio_model.onnx, onnx_meta.json, scaler.joblib, config.json, benchmark_results.csv
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import time
import traceback
import warnings
from pathlib import Path
from types import SimpleNamespace

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

SPLITS = ("train", "val", "test")
ALL_MODELS = ["decision_tree", "svm_rbf", "cnn_custom", "resnet_transfer", "transformer_cnn"]
ONNX_OPSET = 17


# =============================================================== utilitaires
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def peak_rss_mb() -> float:
    """Pic de mémoire RAM du processus (Linux : ru_maxrss en Ko)."""
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:  # Windows
        return float("nan")


def safe_auc(y, p) -> float:
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


def compute_metrics(y: np.ndarray, proba: np.ndarray, prefix: str = "", thr: float = 0.5):
    """Métriques de la classe pathologique (1). Retourne (dict, matrice de confusion)."""
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    recall = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
    try:
        pr_auc = float(average_precision_score(y, proba))
    except ValueError:
        pr_auc = float("nan")
    m = dict(
        recall=recall, specificity=spec, precision=prec, f1=f1,
        accuracy=(tp + tn) / max(len(y), 1), balanced_accuracy=(recall + spec) / 2,
        auc=safe_auc(y, proba), pr_auc=pr_auc,
    )
    return {f"{prefix}{k}": float(v) for k, v in m.items()}, np.array([[tn, fp], [fn, tp]])


def file_level(meta: pd.DataFrame | None, y: np.ndarray, proba: np.ndarray):
    """Agrège les probabilités par fichier .wav (moyenne des segments)."""
    if meta is None or len(meta) != len(y):
        return None
    df = meta[["file", "label"]].copy()
    df["p"] = proba
    g = df.groupby(["file", "label"], as_index=False)["p"].mean()
    return g["label"].to_numpy(), g["p"].to_numpy()


def log_confusion(cm: np.ndarray, title: str, name: str) -> None:
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, int(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["normal", "anormal"])
    ax.set_yticks([0, 1], ["normal", "anormal"])
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title(title)
    fig.tight_layout()
    mlflow.log_figure(fig, name)
    plt.close(fig)


def log_curves(history: list[dict], name: str = "training_curves.png") -> None:
    if not history:
        return
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].plot(h["epoch"], h["train_loss"], label="train")
    ax[0].plot(h["epoch"], h["val_loss"], label="val")
    ax[0].set_title("Perte")
    ax[0].legend()
    ax[1].plot(h["epoch"], h["val_recall"], label="recall")
    ax[1].plot(h["epoch"], h["val_auc"], label="AUC")
    ax[1].set_title("Validation")
    ax[1].legend()
    fig.tight_layout()
    mlflow.log_figure(fig, name)
    plt.close(fig)


# ================================================================== données
def load_data(data_dir: str) -> SimpleNamespace:
    d = Path(data_dir)
    if not (d / "config.json").exists():
        raise SystemExit(f"{d} ne contient pas la sortie de preprocessing.py (config.json absent)")
    X1d = {s: np.load(d / f"X1d_{s}.npy") for s in SPLITS}
    Xmel = {s: np.load(d / f"Xmel_{s}.npy") for s in SPLITS}
    y = {s: np.load(d / f"y_{s}.npy") for s in SPLITS}
    cfg = json.loads((d / "config.json").read_text())

    meta = {s: None for s in SPLITS}
    mp = d / "segments_metadata.csv"
    if mp.exists():
        m = pd.read_csv(mp)
        for s in SPLITS:
            sub = m[m["split"] == s].reset_index(drop=True)
            meta[s] = sub if len(sub) == len(y[s]) else None

    cw = {int(k): float(v) for k, v in cfg["class_weight"].items()}
    mean, std = float(Xmel["train"].mean()), float(Xmel["train"].std())
    return SimpleNamespace(X1d=X1d, Xmel=Xmel, y=y, cfg=cfg, meta=meta, cw=cw,
                           mel_mean=mean, mel_std=std, data_dir=d)


# ============================================================ modèles PyTorch
class MelNorm(nn.Module):
    """Normalisation intégrée au modèle : le .onnx accepte le Mel brut (dB)."""

    def __init__(self, mean: float, std: float):
        super().__init__()
        self.register_buffer("mean", torch.tensor(float(mean)))
        self.register_buffer("std", torch.tensor(float(std)))

    def forward(self, x):
        return (x - self.mean) / (self.std + 1e-6)


def conv_block(cin: int, cout: int, pool=(2, 2)) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(pool),
    )


class CustomCNN(nn.Module):
    def __init__(self, mean, std, drop: float = 0.3):
        super().__init__()
        self.norm = MelNorm(mean, std)
        self.features = nn.Sequential(
            conv_block(1, 16), conv_block(16, 32), conv_block(32, 64), conv_block(64, 128)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(drop), nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.head(self.features(self.norm(x)))


class ResNetTL(nn.Module):
    """ResNet18/50 pré-entraîné ImageNet adapté à 1 canal (poids de conv1 sommés)."""

    def __init__(self, mean, std, depth: int = 18, pretrained: bool = True, drop: float = 0.3):
        super().__init__()
        from torchvision import models

        ctor = getattr(models, f"resnet{depth}")
        net, self.pretrained_loaded = None, False
        if pretrained:
            try:
                net = ctor(weights=getattr(models, f"ResNet{depth}_Weights").IMAGENET1K_V1)
                self.pretrained_loaded = True
            except Exception as e:  # pas de réseau dans le conteneur, etc.
                warnings.warn(f"Poids ImageNet indisponibles ({e}) : ResNet initialisé au hasard")
        if net is None:
            net = ctor(weights=None)
        old = net.conv1
        conv = nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        with torch.no_grad():
            conv.weight.copy_(old.weight.sum(dim=1, keepdim=True))
        net.conv1 = conv
        net.fc = nn.Sequential(nn.Dropout(drop), nn.Linear(net.fc.in_features, 2))
        self.net = net
        self.norm = MelNorm(mean, std)

    def set_backbone_trainable(self, flag: bool) -> None:
        for n, p in self.net.named_parameters():
            if not n.startswith("fc."):
                p.requires_grad = flag

    def forward(self, x):
        return self.net(self.norm(x))


class TransformerCNN(nn.Module):
    """Stem CNN (réduit la fréquence/temps) -> tokens temporels -> encodeur Transformer.
    Le Transformer capte la structure temporelle des bruits (B1/B2, souffles)."""

    def __init__(self, mean, std, d_model=128, heads=4, layers=3, drop=0.2, max_len=512):
        super().__init__()
        self.norm = MelNorm(mean, std)
        self.stem = nn.Sequential(
            conv_block(1, 32, (2, 2)), conv_block(32, 64, (2, 2)), conv_block(64, d_model, (2, 1))
        )
        pos = torch.zeros(max_len, d_model)
        t = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pos[:, 0::2], pos[:, 1::2] = torch.sin(t * div), torch.cos(t * div)
        self.register_buffer("pos", pos)
        layer = nn.TransformerEncoderLayer(
            d_model, heads, dim_feedforward=2 * d_model, dropout=drop,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(d_model, 2))

    def forward(self, x):
        z = self.stem(self.norm(x))                 # (B, C, F', T')
        z = z.mean(dim=2).transpose(1, 2)           # (B, T', C)
        z = z + self.pos[: z.shape[1]]
        z = self.ln(self.encoder(z))
        return self.head(z.mean(dim=1))


class ExportWrapper(nn.Module):
    """Sortie ONNX = probabilités (n, 2), comme les modèles scikit-learn."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.softmax(self.model(x), dim=1)


# ========================================================= entraînement torch
@torch.no_grad()
def predict_torch(model, X: np.ndarray, device, bs: int = 128) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i : i + bs]).to(device)
        out.append(torch.softmax(model(xb), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def eval_loss(model, X, y, crit, device, bs: int = 128) -> tuple[float, np.ndarray]:
    model.eval()
    losses, probs = 0.0, []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i : i + bs]).to(device)
        yb = torch.from_numpy(y[i : i + bs]).to(device)
        logits = model(xb)
        losses += crit(logits, yb).item() * len(yb)
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return losses / len(X), np.concatenate(probs)


def fit_torch(model, ctx, lr: float, freeze_epochs: int = 0):
    a, dev = ctx.args, ctx.device
    Xtr, ytr = ctx.Xmel["train"], ctx.y["train"]
    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=a.batch_size, shuffle=True,
    )
    w = torch.tensor([ctx.cw[0], ctx.cw[1]], dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=w)
    model.to(dev)

    def make_opt(lr_, epochs_left):
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=lr_, weight_decay=1e-4)
        return opt, torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs_left, 1))

    if freeze_epochs > 0 and hasattr(model, "set_backbone_trainable"):
        model.set_backbone_trainable(False)
    opt, sched = make_opt(lr, a.epochs)
    best_score, best_state, bad, history = -1e9, None, 0, []

    for epoch in range(a.epochs):
        if freeze_epochs > 0 and epoch == freeze_epochs:      # dégel du backbone
            model.set_backbone_trainable(True)
            opt, sched = make_opt(lr * 0.1, a.epochs - epoch)
            bad = 0
        model.train()
        run_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_loss += loss.item() * len(yb)
        sched.step()

        vloss, vp = eval_loss(model, ctx.Xmel["val"], ctx.y["val"], crit, dev)
        vm, _ = compute_metrics(ctx.y["val"], vp, "val_")
        rec = dict(epoch=epoch, train_loss=run_loss / len(Xtr), val_loss=vloss,
                   val_recall=vm["val_recall"], val_auc=vm["val_auc"])
        history.append(rec)
        mlflow.log_metrics({k: v for k, v in rec.items() if k != "epoch" and not math.isnan(v)},
                           step=epoch)

        score = vm["val_auc"] if not math.isnan(vm["val_auc"]) else -vloss
        if score > best_score:
            best_score, bad = score, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if bad >= a.patience and epoch >= freeze_epochs:
                print(f"    early stopping à l'époque {epoch}")
                break

    model.load_state_dict(best_state)
    log_curves(history)
    mlflow.log_metric("best_val_auc_epoch_selection", float(best_score))
    mlflow.log_metric("epochs_run", len(history))
    return history


# ====================================================== runners (1 par modèle)
def build_decision_tree(ctx):
    X, y = ctx.X1d, ctx.y
    best, grid = None, []
    for depth in (4, 6, 8, 12, None):
        for leaf in (1, 10):
            clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf,
                                         class_weight=ctx.cw, random_state=ctx.args.seed)
            clf.fit(X["train"], y["train"])
            auc = safe_auc(y["val"], clf.predict_proba(X["val"])[:, 1])
            grid.append(dict(max_depth=depth, min_samples_leaf=leaf, val_auc=auc))
            if best is None or (not math.isnan(auc) and auc > best[0]):
                best = (auc, clf)
    mlflow.log_dict(grid, "grid_search.json")
    clf = best[1]
    params = dict(max_depth=clf.max_depth, min_samples_leaf=clf.min_samples_leaf,
                  class_weight="config.json", n_features=X["train"].shape[1],
                  tuning="grid sur AUC validation")
    return SimpleNamespace(model=clf, params=params, input="x1d", kind="sklearn",
                           predict=lambda Xs: clf.predict_proba(Xs)[:, 1], n_params=clf.tree_.node_count)


def build_svm(ctx):
    X, y = ctx.X1d, ctx.y
    best, grid = None, []
    for C in (0.5, 2.0, 10.0):
        svc = SVC(kernel="rbf", C=C, gamma="scale", class_weight=ctx.cw)
        svc.fit(X["train"], y["train"])
        auc = safe_auc(y["val"], svc.decision_function(X["val"]))
        grid.append(dict(C=C, val_auc=auc))
        if best is None or (not math.isnan(auc) and auc > best[0]):
            best = (auc, C)
    mlflow.log_dict(grid, "grid_search.json")
    # refit final avec probabilités (Platt) : nécessaire pour ONNX et le seuil 0.5
    svc = SVC(kernel="rbf", C=best[1], gamma="scale", class_weight=ctx.cw,
              probability=True, random_state=ctx.args.seed)
    svc.fit(X["train"], y["train"])
    params = dict(kernel="rbf", C=best[1], gamma="scale", probability=True,
                  class_weight="config.json", tuning="grid C sur AUC validation")
    return SimpleNamespace(model=svc, params=params, input="x1d", kind="sklearn",
                           predict=lambda Xs: svc.predict_proba(Xs)[:, 1],
                           n_params=int(svc.support_vectors_.size))


def _torch_result(model, ctx, params, lr, freeze_epochs=0):
    if ctx.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    n_params = sum(p.numel() for p in model.parameters())
    fit_torch(model, ctx, lr, freeze_epochs)
    if ctx.device.type == "cuda":
        mlflow.log_metric("peak_vram_mb", torch.cuda.max_memory_allocated() / 1e6)
    params.update(lr=lr, batch_size=ctx.args.batch_size, epochs_max=ctx.args.epochs,
                  patience=ctx.args.patience, optimizer="AdamW", loss="CrossEntropy pondérée",
                  trainable_params=n_params, device=str(ctx.device))
    dev = ctx.device
    return SimpleNamespace(model=model, params=params, input="xmel", kind="torch",
                           predict=lambda Xs: predict_torch(model, Xs, dev), n_params=n_params)


def build_cnn(ctx):
    model = CustomCNN(ctx.mel_mean, ctx.mel_std)
    return _torch_result(model, ctx, dict(arch="CNN 4 blocs conv 16-32-64-128"), ctx.args.lr)


def build_resnet(ctx):
    model = ResNetTL(ctx.mel_mean, ctx.mel_std, ctx.args.resnet_depth, not ctx.args.no_pretrained)
    params = dict(arch=f"ResNet{ctx.args.resnet_depth}", pretrained=model.pretrained_loaded,
                  freeze_epochs=ctx.args.freeze_epochs)
    return _torch_result(model, ctx, params, ctx.args.lr, freeze_epochs=ctx.args.freeze_epochs)


def build_transformer(ctx):
    model = TransformerCNN(ctx.mel_mean, ctx.mel_std)
    params = dict(arch="CNN stem + TransformerEncoder (3 couches, 4 têtes, d=128)")
    return _torch_result(model, ctx, params, ctx.args.lr * 0.5)


BUILDERS = {
    "decision_tree": ("classique", build_decision_tree),
    "svm_rbf": ("classique", build_svm),
    "cnn_custom": ("deep", build_cnn),
    "resnet_transfer": ("deep", build_resnet),
    "transformer_cnn": ("deep", build_transformer),
}


# ===================================================== évaluation + journal
def evaluate_and_log(res, ctx) -> dict:
    Xs = ctx.X1d if res.input == "x1d" else ctx.Xmel
    out = {}
    for split in ("val", "test"):
        y = ctx.y[split]
        t0 = time.perf_counter()
        p = res.predict(Xs[split])
        if split == "val":
            out["latency_ms_per_segment"] = (time.perf_counter() - t0) / len(y) * 1000
        m, cm = compute_metrics(y, p, f"{split}_")
        out.update(m)
        fl = file_level(ctx.meta[split], y, p)
        if fl is not None:
            fm, _ = compute_metrics(fl[0], fl[1], f"{split}_file_")
            out.update(fm)
        if split == "test":
            log_confusion(cm, "Test (segments)", "confusion_matrix_test.png")
    out["n_params"] = float(res.n_params)
    out["peak_rss_mb"] = peak_rss_mb()
    mlflow.log_metrics({k: v for k, v in out.items() if not (isinstance(v, float) and math.isnan(v))})
    return out


def execute(name: str, ctx) -> dict | None:
    family, builder = BUILDERS[name]
    print(f"\n=== {name} ({family}) ===")
    try:
        with mlflow.start_run(run_name=name, nested=True) as run:
            mlflow.set_tags({"algo": name, "family": family})
            t0 = time.perf_counter()
            res = builder(ctx)
            train_time = time.perf_counter() - t0
            mlflow.log_params(res.params)
            mlflow.log_param("input", res.input)
            mlflow.log_metric("train_time_s", train_time)
            metrics = evaluate_and_log(res, ctx)
            ctx.fitted[name] = res
            row = dict(model=name, run_id=run.info.run_id, input=res.input,
                       train_time_s=train_time, **metrics)
            print(f"    val recall={row['val_recall']:.3f} spec={row['val_specificity']:.3f} "
                  f"auc={row['val_auc']:.3f} | test recall={row['test_recall']:.3f} "
                  f"({train_time:.0f}s)")
            return row
    except Exception:
        print(f"    ECHEC de {name} (run marqué FAILED) :")
        traceback.print_exc()
        return None


# ============================================================ sélection / ONNX
def rank_models(rows: list[dict], min_spec: float) -> list[dict]:
    """Maximise le RECALL de validation (classe pathologique), sous contrainte de
    spécificité minimale (évite le modèle triviale « tout anormal »), puis F1, AUC."""
    def key(r):
        auc = 0.0 if math.isnan(r["val_auc"]) else r["val_auc"]
        return (r["val_specificity"] >= min_spec, r["val_recall"], r["val_f1"], auc)

    return sorted(rows, key=key, reverse=True)


def export_onnx(res, ctx, path: Path) -> None:
    if res.kind == "sklearn":
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        n_feat = ctx.X1d["train"].shape[1]
        onx = convert_sklearn(
            res.model, initial_types=[("x1d", FloatTensorType([None, n_feat]))],
            options={id(res.model): {"zipmap": False}}, target_opset=15,
        )
        path.write_bytes(onx.SerializeToString())
    else:
        model = res.model.to("cpu").eval()
        shape = (2,) + tuple(ctx.Xmel["train"].shape[1:])
        kwargs = dict(input_names=["xmel"], output_names=["probabilities"],
                      dynamic_axes={"xmel": {0: "batch"}, "probabilities": {0: "batch"}},
                      opset_version=ONNX_OPSET)
        dummy = torch.randn(*shape)
        wrapper = ExportWrapper(model).eval()
        try:
            torch.onnx.export(wrapper, dummy, str(path), dynamo=False, **kwargs)
        except TypeError:  # torch < 2.5
            torch.onnx.export(wrapper, dummy, str(path), **kwargs)


def verify_onnx(res, ctx, path: Path) -> dict:
    """Recharge le .onnx avec ONNX Runtime et compare avec le modèle d'origine."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    idx = out_names.index("probabilities") if "probabilities" in out_names else len(out_names) - 1
    X = (ctx.X1d if res.input == "x1d" else ctx.Xmel)["test"]
    y = ctx.y["test"]

    p_onnx = np.concatenate([sess.run(None, {in_name: X[i : i + 128]})[idx][:, 1]
                             for i in range(0, len(X), 128)])
    ref = (res.model.predict_proba(X)[:, 1] if res.kind == "sklearn"
           else predict_torch(res.model.to("cpu"), X, torch.device("cpu")))
    max_diff = float(np.abs(p_onnx - ref).max())
    if max_diff > 1e-2:
        raise RuntimeError(f"ONNX diverge du modèle d'origine (écart max {max_diff:.4f})")

    n = min(256, len(X))
    t0 = time.perf_counter()
    sess.run(None, {in_name: X[:n]})
    lat = (time.perf_counter() - t0) / n * 1000
    m, _ = compute_metrics(y, p_onnx, "onnx_test_")
    return dict(input_name=in_name, output_name=out_names[idx], max_abs_diff=max_diff,
                onnx_latency_ms_per_segment=lat, **m)


def export_best(ranked: list[dict], ctx, out_dir: Path):
    """Exporte le meilleur modèle ; si l'export échoue, passe au suivant."""
    for row in ranked:
        res = ctx.fitted[row["model"]]
        path = out_dir / "cardio_model.onnx"
        try:
            export_onnx(res, ctx, path)
            chk = verify_onnx(res, ctx, path)
            return row, res, path, chk
        except Exception:
            print(f"Export ONNX impossible pour {row['model']}, essai du suivant :")
            traceback.print_exc()
    raise SystemExit("Aucun modèle n'a pu être exporté en ONNX")


def register_model(path: Path, name: str, row: dict) -> None:
    try:
        import onnx

        mlflow.onnx.log_model(onnx.load(str(path)), artifact_path="onnx_model",
                              registered_model_name=name)
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{name}'")
        ver = str(max(int(v.version) for v in versions))
        client.set_registered_model_alias(name, "champion", ver)
        client.set_model_version_tag(name, ver, "algo", row["model"])
        client.set_model_version_tag(name, ver, "val_recall", f"{row['val_recall']:.4f}")
        client.set_model_version_tag(name, ver, "test_recall", f"{row['test_recall']:.4f}")
        print(f"Model Registry : {name} v{ver} (alias 'champion')")
    except Exception as e:
        warnings.warn(f"Enregistrement dans le Model Registry impossible : {e}")


# ====================================================================== main
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out-dir", default="audio_engine/models")
    ap.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--resnet-depth", type=int, default=18, choices=[18, 50])
    ap.add_argument("--freeze-epochs", type=int, default=3,
                    help="époques avec backbone ResNet gelé (tête seule)")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--min-specificity", type=float, default=0.5,
                    help="spécificité de validation minimale pour être éligible")
    ap.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    ap.add_argument("--experiment", default="cardio-audio-benchmark")
    ap.add_argument("--registered-name", default="cardio-pcg-classifier")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = load_data(args.data_dir)
    ctx.args, ctx.seed, ctx.fitted = args, args.seed, {}
    ctx.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    print(f"MLflow : {args.tracking_uri} | device : {ctx.device}")

    with mlflow.start_run(run_name="benchmark") as parent:
        sizes = {f"n_{s}": int(len(ctx.y[s])) for s in SPLITS}
        pos = {f"pct_abnormal_{s}": float(ctx.y[s].mean()) for s in SPLITS}
        mlflow.log_params({**sizes, **{k: round(v, 4) for k, v in pos.items()},
                           **{k: v for k, v in ctx.cfg.items() if k != "class_weight"},
                           "class_weight": json.dumps(ctx.cfg["class_weight"]),
                           "seed": args.seed, "device": str(ctx.device),
                           "selection_rule": f"max val recall (spec >= {args.min_specificity})",
                           "models": ",".join(args.models)})
        mlflow.log_artifact(str(ctx.data_dir / "config.json"), "preprocessing")
        if (ctx.data_dir / "scaler.joblib").exists():
            mlflow.log_artifact(str(ctx.data_dir / "scaler.joblib"), "preprocessing")

        rows = [r for r in (execute(m, ctx) for m in args.models) if r is not None]
        if not rows:
            raise SystemExit("Aucun modèle entraîné avec succès")

        ranked = rank_models(rows, args.min_specificity)
        if ranked[0]["val_specificity"] < args.min_specificity:
            warnings.warn("Aucun modèle ne respecte la spécificité minimale : "
                          "sélection sur le recall seul")

        df = pd.DataFrame(ranked)
        df.to_csv(out_dir / "benchmark_results.csv", index=False)
        mlflow.log_artifact(str(out_dir / "benchmark_results.csv"))
        cols = ["model", "val_recall", "val_specificity", "val_f1", "val_auc",
                "test_recall", "test_specificity", "test_f1", "test_auc", "train_time_s"]
        print("\n" + df[cols].round(3).to_string(index=False))

        best_row, best_res, onnx_path, chk = export_best(ranked, ctx, out_dir)

        meta = dict(
            algo=best_row["model"], input_kind=best_res.input, input_name=chk["input_name"],
            output_name=chk["output_name"], positive_class=1, threshold=0.5,
            input_shape=([None, int(ctx.X1d["train"].shape[1])] if best_res.input == "x1d"
                         else [None] + [int(s) for s in ctx.Xmel["train"].shape[1:]]),
            val_recall=best_row["val_recall"], test_recall=best_row["test_recall"],
            mlflow_run_id=parent.info.run_id,
        )
        (out_dir / "onnx_meta.json").write_text(json.dumps(meta, indent=2))
        shutil.copy(ctx.data_dir / "config.json", out_dir / "config.json")
        if (ctx.data_dir / "scaler.joblib").exists():
            shutil.copy(ctx.data_dir / "scaler.joblib", out_dir / "scaler.joblib")

        mlflow.set_tags({"best_model": best_row["model"], "best_model_run_id": best_row["run_id"]})
        mlflow.log_metrics({"best_val_recall": best_row["val_recall"],
                            "best_test_recall": best_row["test_recall"],
                            "onnx_max_abs_diff": chk["max_abs_diff"],
                            "onnx_latency_ms_per_segment": chk["onnx_latency_ms_per_segment"],
                            "onnx_test_recall": chk["onnx_test_recall"],
                            "onnx_size_mb": onnx_path.stat().st_size / 1e6})
        mlflow.log_artifact(str(out_dir / "onnx_meta.json"), "best_model")
        register_model(onnx_path, args.registered_name, best_row)

        print(f"\nMeilleur modèle : {best_row['model']} "
              f"(recall val={best_row['val_recall']:.3f}, test={best_row['test_recall']:.3f})")
        print(f"ONNX : {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} Mo, "
              f"écart max vs original {chk['max_abs_diff']:.2e})")
        print("Le recall de TEST est l'estimation honnête : la sélection s'est faite sur la validation.")


if __name__ == "__main__":
    main()