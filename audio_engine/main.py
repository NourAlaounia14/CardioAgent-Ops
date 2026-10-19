"""audio_engine/main.py : service 3 (audio-engine), FastAPI.

Lancement local :
    uvicorn audio_engine.main:app --host 0.0.0.0 --port 8000
Appelé par l'AudioAgent : POST http://audio-engine:8000/predict (champ "file").
"""
import os
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile

from audio_engine.preprocessing import preprocess_bytes

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/cardio_model.onnx"))
SCALER_PATH = Path(os.getenv("SCALER_PATH", "data/processed/scaler.joblib"))
MODEL_INPUT = os.getenv("MODEL_INPUT", "mel")      # "mel" (CNN/ResNet/...) ou "1d" (SVM/arbre)
MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
CLASSES = ["normal", "anormal"]

app = FastAPI(title="audio-engine")
state = {"session": None, "scaler": None}


@app.on_event("startup")
def load_artifacts():
    if SCALER_PATH.exists():
        state["scaler"] = joblib.load(SCALER_PATH)
    if MODEL_PATH.exists():
        state["session"] = ort.InferenceSession(
            str(MODEL_PATH), providers=["CPUExecutionProvider"]
        )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["session"] is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if state["session"] is None or state["scaler"] is None:
        raise HTTPException(503, "Modèle ONNX ou scaler non chargé")
    if not (file.filename or "").lower().endswith(".wav"):
        raise HTTPException(415, "Seuls les fichiers .wav sont acceptés")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Fichier trop volumineux")
    try:
        x1d, xmel = preprocess_bytes(data, state["scaler"])
    except Exception as e:
        raise HTTPException(422, f"Audio illisible : {e}")

    x = xmel if MODEL_INPUT == "mel" else x1d
    sess = state["session"]
    probs = sess.run(None, {sess.get_inputs()[0].name: x})[0]   # attendu : (n_segments, 2)
    # Agrégation sur les segments : la moyenne des probabilités
    # (pour favoriser le Recall tu peux utiliser le max de la proba "anormal").
    mean = probs.mean(axis=0)
    idx = int(mean.argmax())
    return {
        "classe": CLASSES[idx],
        "confiance": round(float(mean[idx]), 4),
        "n_segments": int(len(probs)),
    }