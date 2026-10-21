"""Clients vers vector-db (Service 2) et llm-server (Service 1), avec mode mock.

USE_MOCKS=true  -> aucune dépendance réseau (développement / tests / CI).
USE_MOCKS=false -> vrais appels (Qdrant REST + Ollama).
"""
import json
import os
from functools import lru_cache

import httpx

TIMEOUT = httpx.Timeout(120.0, connect=5.0)   # un LLM 7B sur CPU peut être lent

MOCK_PASSAGES = [
    {"source": "ESC Guidelines 2021 - Valvular Heart Disease",
     "excerpt": "Un souffle systolique associé à des symptômes impose une échocardiographie transthoracique.",
     "similarity_score": 0.91},
    {"source": "ESC Guidelines 2021 - Heart Failure",
     "excerpt": "Toute suspicion de dysfonction cardiaque doit conduire à une évaluation cardiologique spécialisée.",
     "similarity_score": 0.84},
    {"source": "ESC Guidelines 2021 - Cardiovascular Prevention",
     "excerpt": "En l'absence d'anomalie, un suivi clinique de routine est suffisant.",
     "similarity_score": 0.62},
]


def _mocks() -> bool:
    return os.getenv("USE_MOCKS", "true").lower() == "true"


@lru_cache(maxsize=1)
def _embedder():
    # Import tardif : sentence-transformers est lourd, inutile en mode mock.
    # Le modèle DOIT être le même que celui utilisé par rag_engine/ pour l'indexation.
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))


def retrieve_guidelines(query: str, k: int = 3) -> list[dict]:
    if _mocks():
        return MOCK_PASSAGES[:k]
    url = os.getenv("VECTOR_DB_URL", "http://vector-db:6333")
    collection = os.getenv("QDRANT_COLLECTION", "esc_guidelines")
    vector = _embedder().encode(query).tolist()
    r = httpx.post(f"{url}/collections/{collection}/points/search",
                   json={"vector": vector, "limit": k, "with_payload": True}, timeout=TIMEOUT)
    r.raise_for_status()
    out = []
    for hit in r.json()["result"]:
        payload = hit.get("payload", {})
        # ADAPTER 'source' / 'text' aux noms de champs utilisés par rag_engine/
        out.append({"source": payload.get("source", "ESC"),
                    "excerpt": payload.get("text", ""),
                    "similarity_score": round(float(hit["score"]), 4)})
    return out


def _mock_llm(ctx: dict) -> str:
    p = ctx.get("p_abnormal", 0.0)
    sources = ctx.get("sources", [])
    if p >= 0.8:
        risk, text = "high", "Signal fortement évocateur d'une anomalie cardiaque (souffle probable). Avis cardiologique rapide recommandé."
    elif p >= 0.5:
        risk, text = "moderate", "Signal suspect d'une anomalie cardiaque. Une évaluation cardiologique est recommandée."
    else:
        risk, text = "low", "Signal sans anomalie détectée. Un suivi clinique de routine est suffisant."
    cited = sources[:1] if p >= 0.5 else []
    return json.dumps({"diagnosis": text, "risk_level": risk, "cited_sources": cited}, ensure_ascii=False)


def call_llm(messages: list[dict], mock_context: dict | None = None) -> str:
    if _mocks():
        return _mock_llm(mock_context or {})
    url = os.getenv("LLM_SERVER_URL", "http://llm-server:11434")
    r = httpx.post(f"{url}/api/chat", timeout=TIMEOUT, json={
        "model": os.getenv("LLM_MODEL", "cardio-qwen"),
        "messages": messages, "stream": False, "format": "json",
        "options": {"temperature": 0.2},
    })
    r.raise_for_status()
    return r.json()["message"]["content"]
