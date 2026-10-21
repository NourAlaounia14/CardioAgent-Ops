"""
Service 5 : api-gateway
Point d'entrée unique de la plateforme CardioAgent-Ops.
Reçoit le fichier .wav du client, orchestre les appels vers
audio-engine (Service 3) puis agent-orchestrator (Service 4),
et renvoie le rapport JSON structuré (AnalysisReport).
"""

import os
import uuid
import logging
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from schemas import AnalysisReport, ErrorResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api-gateway")

app = FastAPI(
    title="CardioAgent-Ops API Gateway",
    description="Point d'entrée unique — endpoint /analyze_pcg",
    version="1.0.0",
)

# Autorise le client (fichier HTML local, "null" origin, ou servi via un autre
# port) à appeler ce gateway depuis le navigateur. En développement on accepte
# large ("*") ; à resserrer en production sur l'origine réelle du client.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---------------------------------------------------------
# Noms de service Docker (docker-network) : utilisés comme hostnames une fois
# le docker-compose.yml en place. En local (hors Docker), on peut surcharger
# ces valeurs via des variables d'environnement.
AUDIO_ENGINE_URL = os.getenv("AUDIO_ENGINE_URL", "http://audio-engine:8000")
AGENT_ORCHESTRATOR_URL = os.getenv("AGENT_ORCHESTRATOR_URL", "http://agent-orchestrator:8000")

# Pendant que les Services 3 et 4 ne sont pas encore prêts, on peut basculer
# sur des réponses simulées pour développer et tester le gateway seul.
USE_MOCKS = os.getenv("USE_MOCKS", "true").lower() == "true"

ALLOWED_EXTENSIONS = {".wav"}
MAX_FILE_SIZE_MB = 20

HTTP_TIMEOUT = httpx.Timeout(30.0, connect=5.0)
# L'orchestrateur appelle un LLM 7B (potentiellement sur CPU) : timeout bien plus long.
ORCHESTRATOR_TIMEOUT = httpx.Timeout(180.0, connect=5.0)


# --- Endpoints ---------------------------------------------------------

@app.get("/health")
async def health():
    """Vérification rapide que le gateway est en vie (utile pour Docker healthcheck / CI)."""
    return {"status": "ok"}


@app.post(
    "/analyze_pcg",
    response_model=AnalysisReport,
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def analyze_pcg(file: UploadFile = File(...)):
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Nouvelle requête reçue : {file.filename}")

    # 1. Validation du fichier
    _validate_wav_file(file)
    audio_bytes = await file.read()
    _validate_file_size(audio_bytes)

    # 2. Appel audio-engine (Service 3)
    audio_features = await _call_audio_engine(request_id, file.filename, audio_bytes)

    # 3. Appel agent-orchestrator (Service 4)
    report = await _call_agent_orchestrator(request_id, audio_features)

    return report


# --- Validation ---------------------------------------------------------

def _validate_wav_file(file: UploadFile) -> None:
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Format non supporté ({ext}). Seul .wav est accepté.",
        )


def _validate_file_size(audio_bytes: bytes) -> None:
    size_mb = len(audio_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=422,
            detail=f"Fichier trop volumineux ({size_mb:.1f} Mo, max {MAX_FILE_SIZE_MB} Mo).",
        )
    if size_mb == 0:
        raise HTTPException(status_code=422, detail="Fichier vide.")


# --- Appels aux services internes ---------------------------------------

def _normalize_audio_result(raw: dict) -> dict:
    """Accepte la réponse d'audio-engine en français (classe/confiance) ou en
    anglais (predicted_class/confidence) et la ramène au format AudioFeatures.
    Si 'probabilities' est absent (classification binaire), on le déduit."""
    predicted = raw.get("predicted_class") or raw.get("classe")
    confidence = raw.get("confidence", raw.get("confiance"))
    if predicted is None or confidence is None:
        raise HTTPException(status_code=502, detail="Réponse audio-engine invalide.")
    confidence = float(confidence)
    probs = raw.get("probabilities")
    if not probs:
        other = "normal" if predicted == "anormal" else "anormal"
        probs = {predicted: confidence, other: round(1 - confidence, 4)}
    return {"predicted_class": predicted, "confidence": confidence, "probabilities": probs}


async def _call_audio_engine(request_id: str, filename: str, audio_bytes: bytes) -> dict:
    if USE_MOCKS:
        logger.info(f"[{request_id}] MOCK audio-engine")
        return {
            "predicted_class": "anormal",
            "confidence": 0.87,
            "probabilities": {"normal": 0.13, "anormal": 0.87},
        }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.post(
                f"{AUDIO_ENGINE_URL}/predict",
                files={"file": (filename, audio_bytes, "audio/wav")},
            )
            response.raise_for_status()
            return _normalize_audio_result(response.json())
    except httpx.ConnectError:
        logger.error(f"[{request_id}] audio-engine injoignable")
        raise HTTPException(status_code=503, detail="Service audio-engine indisponible.")
    except httpx.TimeoutException:
        logger.error(f"[{request_id}] Timeout audio-engine")
        raise HTTPException(status_code=503, detail="Timeout audio-engine.")
    except httpx.HTTPStatusError as exc:
        logger.error(f"[{request_id}] Erreur audio-engine : {exc}")
        raise HTTPException(status_code=502, detail="Erreur retournée par audio-engine.")


async def _call_agent_orchestrator(request_id: str, audio_features: dict) -> dict:
    if USE_MOCKS:
        logger.info(f"[{request_id}] MOCK agent-orchestrator")
        return {
            "request_id": request_id,
            "diagnosis": "Souffle cardiaque suspect — orientation cardiologique recommandée.",
            "confidence": audio_features["confidence"],
            "risk_level": "moderate",
            "audio_features": audio_features,
            "clinical_recommendations": [
                {
                    "source": "ESC Guidelines 2023 — Valvular Heart Disease",
                    "excerpt": "Recommandation simulée pour le développement.",
                    "similarity_score": 0.91,
                }
            ],
            "audit_notes": "Réponse simulée (USE_MOCKS=true).",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    try:
        async with httpx.AsyncClient(timeout=ORCHESTRATOR_TIMEOUT) as client:
            response = await client.post(
                f"{AGENT_ORCHESTRATOR_URL}/orchestrate",
                json={"request_id": request_id, "audio_features": audio_features},
            )
            response.raise_for_status()
            return response.json()
    except httpx.ConnectError:
        logger.error(f"[{request_id}] agent-orchestrator injoignable")
        raise HTTPException(status_code=503, detail="Service agent-orchestrator indisponible.")
    except httpx.TimeoutException:
        logger.error(f"[{request_id}] Timeout agent-orchestrator")
        raise HTTPException(status_code=503, detail="Timeout agent-orchestrator.")
    except httpx.HTTPStatusError as exc:
        logger.error(f"[{request_id}] Erreur agent-orchestrator : {exc}")
        raise HTTPException(status_code=502, detail="Erreur retournée par agent-orchestrator.")


# --- Gestion d'erreurs globale ---------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception("Erreur interne non gérée")
    return JSONResponse(
        status_code=500,
        content={"detail": "Erreur interne du serveur.", "error_code": "INTERNAL_ERROR"},
    )
