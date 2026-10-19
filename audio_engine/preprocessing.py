"""audio_engine/preprocessing.py

Prétraitement commun aux 5 algorithmes ET au service FastAPI (service 3).
Les fonctions d'entraînement et d'inférence sont les MÊMES : aucune dérive
possible entre l'entraînement et la production.

Usage (entraînement) :
    python -m audio_engine.preprocessing --data-root data/raw --out-dir data/processed

Structure attendue :
    data/raw/normal/*.wav
    data/raw/abnormal/*.wav      (les dossiers peuvent être imbriqués)
Label : 0 = normal, 1 = anormal (classe pathologique).
"""
from __future__ import annotations

import argparse
import io
import json
import re
import warnings
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfiltfilt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------- paramètres
SR = 2000                  # fréquence cible (Hz)
SEG_SEC = 5                # durée d'un segment (2 ou 5)
SEG_LEN = SR * SEG_SEC
HOP_SEC = SEG_SEC / 2      # chevauchement de 50 %
BAND = (25, 400)           # bande utile des sons cardiaques (Hz)
MIN_SEC = 1.0              # fichiers plus courts : ignorés
N_MFCC = 20
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 32            # -> 313 trames pour 5 s
SEED = 42


# ------------------------------------------------------------- 1. inventaire
def patient_id_from_name(stem: str) -> str:
    """Déduit l'identifiant patient du nom de fichier.

    Par défaut : on retire un suffixe d'index de segment (_1, _02...).
    ADAPTE cette fonction à ton dataset si plusieurs fichiers = même patient
    (ex. PASCAL : le numéro avant le timestamp).
    Si rien ne permet de regrouper, chaque fichier = un patient.
    """
    return re.sub(r"[_\-]\d+$", "", stem)


def build_inventory(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    rows = []
    for p in sorted(root.rglob("*.wav")):
        parts = [x.lower() for x in p.relative_to(root).parts[:-1]]
        if any("abnormal" in x or "anormal" in x for x in parts):
            label = 1
        elif any("normal" in x for x in parts):
            label = 0
        else:
            warnings.warn(f"Label introuvable pour {p}, fichier ignoré")
            continue
        try:
            info = sf.info(str(p))
        except RuntimeError:
            warnings.warn(f"Fichier corrompu ignoré : {p}")
            continue
        rows.append(
            dict(
                path=str(p),
                file=p.name,
                patient_id=patient_id_from_name(p.stem),
                label=label,
                orig_sr=info.samplerate,
                channels=info.channels,
                duration=info.frames / info.samplerate,
            )
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"Aucun .wav trouvé dans {root}")
    return df


# ------------------------------------------- 2-6. signal : chargement -> segments
def load_signal(source) -> np.ndarray:
    """Charge en mono et rééchantillonne à SR (chemin ou objet fichier)."""
    y, _ = librosa.load(source, sr=SR, mono=True)
    return y


def bandpass(y: np.ndarray, band=BAND, order=4) -> np.ndarray:
    sos = butter(order, band, btype="bandpass", fs=SR, output="sos")
    return sosfiltfilt(sos, y).astype(np.float32)


def normalize(y: np.ndarray) -> np.ndarray:
    """Normalisation d'amplitude (pic = 1) : on ignore le volume d'enregistrement."""
    return (y / (np.max(np.abs(y)) + 1e-8)).astype(np.float32)


def segment(y: np.ndarray) -> np.ndarray:
    """Découpe en segments de SEG_LEN (chevauchement 50 %), padding si trop court."""
    if len(y) < SEG_LEN:
        return np.pad(y, (0, SEG_LEN - len(y)))[None, :]
    hop = int(HOP_SEC * SR)
    starts = list(range(0, len(y) - SEG_LEN + 1, hop))
    if starts[-1] + SEG_LEN < len(y):          # ne pas perdre la fin du signal
        starts.append(len(y) - SEG_LEN)
    return np.stack([y[s : s + SEG_LEN] for s in starts])


def signal_to_segments(source) -> np.ndarray:
    return segment(normalize(bandpass(load_signal(source))))


# --------------------------------------------------------------- 7. features
def features_1d(seg: np.ndarray) -> np.ndarray:
    """Vecteur 1D (42 valeurs) pour Decision Tree et SVM."""
    mfcc = librosa.feature.mfcc(
        y=seg, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=40, fmin=BAND[0], fmax=BAND[1],
    )
    zcr = librosa.feature.zero_crossing_rate(
        seg, frame_length=N_FFT, hop_length=HOP_LENGTH
    )
    return np.concatenate(
        [mfcc.mean(1), mfcc.std(1), zcr.mean(1), zcr.std(1)]
    ).astype(np.float32)


def mel_spectrogram(seg: np.ndarray) -> np.ndarray:
    """Image 2D (N_MELS x 313) en dB pour CNN, ResNet, Transformer."""
    m = librosa.feature.melspectrogram(
        y=seg, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
        fmin=BAND[0], fmax=BAND[1],
    )
    return librosa.power_to_db(m, ref=np.max).astype(np.float32)


# ------------------------------------------------- 9. augmentation (train seul)
def augment(seg: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shifted = np.roll(seg, int(rng.integers(-SR // 4, SR // 4)))
    noisy = shifted + rng.normal(0, 0.005, size=shifted.shape)
    return normalize(noisy.astype(np.float32))


# ------------------------------------------------------ 8. split par patient
def split_by_patient(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """70/15/15, tous les fichiers d'un patient dans UN SEUL groupe.
    Fait AVANT la segmentation."""
    df = df.reset_index(drop=True).copy()
    g1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, rest_idx = next(g1.split(df, groups=df["patient_id"]))
    rest = df.iloc[rest_idx]
    g2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_rel, test_rel = next(g2.split(rest, groups=rest["patient_id"]))
    df["split"] = ""
    df.loc[train_idx, "split"] = "train"
    df.loc[rest.index[val_rel], "split"] = "val"
    df.loc[rest.index[test_rel], "split"] = "test"
    return df


# ------------------------------------------------------------- 10. pipeline
def run_pipeline(data_root: str, out_dir: str, augment_abnormal: bool = True):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    inv = build_inventory(data_root)
    print(f"{len(inv)} fichiers | {inv['patient_id'].nunique()} patients")
    print(inv["label"].value_counts().rename({0: "normal", 1: "anormal"}).to_string())
    inv = inv[inv["duration"] >= MIN_SEC]
    inv = split_by_patient(inv)
    inv.to_csv(out / "inventory.csv", index=False)

    X1, XM, meta = {s: [] for s in ("train", "val", "test")}, {s: [] for s in ("train", "val", "test")}, []
    y_split = {s: [] for s in ("train", "val", "test")}

    for _, row in inv.iterrows():
        s = row["split"]
        segs = signal_to_segments(row["path"])
        for i, seg in enumerate(segs):
            variants = [(seg, 0)]
            if s == "train" and augment_abnormal and row["label"] == 1:
                variants.append((augment(seg, rng), 1))
            for sig, is_aug in variants:
                X1[s].append(features_1d(sig))
                XM[s].append(mel_spectrogram(sig))
                y_split[s].append(row["label"])
                meta.append(dict(split=s, file=row["file"], patient_id=row["patient_id"],
                                 label=row["label"], seg=i, augmented=is_aug))

    scaler = StandardScaler().fit(np.stack(X1["train"]))      # train SEULEMENT
    for s in ("train", "val", "test"):
        np.save(out / f"X1d_{s}.npy", scaler.transform(np.stack(X1[s])).astype(np.float32))
        np.save(out / f"Xmel_{s}.npy", np.stack(XM[s])[:, None, :, :])  # (n,1,64,313)
        np.save(out / f"y_{s}.npy", np.array(y_split[s], dtype=np.int64))
    joblib.dump(scaler, out / "scaler.joblib")
    pd.DataFrame(meta).to_csv(out / "segments_metadata.csv", index=False)

    y_tr = np.array(y_split["train"])
    n_neg, n_pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
    config = dict(SR=SR, SEG_SEC=SEG_SEC, HOP_SEC=HOP_SEC, BAND=BAND, N_MFCC=N_MFCC,
                  N_MELS=N_MELS, N_FFT=N_FFT, HOP_LENGTH=HOP_LENGTH,
                  class_weight={"0": len(y_tr) / (2 * max(n_neg, 1)),
                                "1": len(y_tr) / (2 * max(n_pos, 1))})
    (out / "config.json").write_text(json.dumps(config, indent=2))
    for s in ("train", "val", "test"):
        ys = np.array(y_split[s])
        print(f"{s:5s}: {len(ys):5d} segments | anormaux = {ys.mean():.1%}")


# ------------------------------------------------- inférence (service FastAPI)
def preprocess_bytes(data: bytes, scaler: StandardScaler):
    """Mêmes transformations qu'à l'entraînement, à partir des octets d'un .wav."""
    segs = signal_to_segments(io.BytesIO(data))
    x1d = scaler.transform(np.stack([features_1d(s) for s in segs])).astype(np.float32)
    xmel = np.stack([mel_spectrogram(s) for s in segs])[:, None, :, :]
    return x1d, xmel


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--no-augment", action="store_true")
    a = ap.parse_args()
    run_pipeline(a.data_root, a.out_dir, augment_abnormal=not a.no_augment)