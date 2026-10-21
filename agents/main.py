"""Service 4 : agent-orchestrator. POST /orchestrate -> rapport validé (AnalysisReport)."""
import logging

import httpx
from fastapi import FastAPI, HTTPException

from graph import run
from schemas import AnalysisReport, OrchestrateRequest
from tracking import log_run

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent-orchestrator")

app = FastAPI(title="CardioAgent-Ops Agent Orchestrator", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/orchestrate", response_model=AnalysisReport)
def orchestrate(req: OrchestrateRequest):
    logger.info("[%s] orchestration démarrée", req.request_id)
    try:
        final = run(req.request_id, req.audio_features.model_dump())
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        logger.error("[%s] service en aval indisponible : %s", req.request_id, e)
        raise HTTPException(503, "llm-server ou vector-db indisponible.")
    except httpx.HTTPStatusError as e:
        logger.error("[%s] erreur service en aval : %s", req.request_id, e)
        raise HTTPException(502, "Erreur retournée par llm-server ou vector-db.")
    log_run(final)
    return final["report"]
