"""rag_engine/api.py : API FastAPI du RAG (utilisable seule, ou appelée par le KnowledgeRAG-Agent).

Lancement :   uvicorn rag_engine.api:app --host 0.0.0.0 --port 8001
Test :        curl -X POST http://localhost:8001/search -H "Content-Type: application/json" \
                   -d '{"query": "souffle systolique, conduite à tenir", "k": 3, "mode": "hybrid"}'
"""
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_engine import config
from rag_engine.Bm25 import get_index
from rag_engine.embedder import get_embedder
from rag_engine.search import _client, format_context, search

app = FastAPI(title="rag-engine")


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    k: int = Field(4, ge=1, le=10)
    source: str | None = None
    mode: Literal["hybrid", "dense", "bm25"] = "hybrid"
    min_score: float = Field(0.0, ge=0.0, le=1.0)      # mode "dense" uniquement (similarité cosinus)


class Hit(BaseModel):
    score: float            # dense : cosinus | bm25 : score BM25 | hybrid : score de fusion RRF
    text: str
    heading: str
    source: str
    page: int | None = None
    chunk_id: str
    # détail renseigné en mode "hybrid" (None si le chunk n'est apparu que dans une des deux listes)
    dense_score: float | None = None
    bm25_score: float | None = None
    dense_rank: int | None = None
    bm25_rank: int | None = None


class SearchResponse(BaseModel):
    mode: str
    hits: list[Hit]
    context: str            # bloc prêt à insérer dans le prompt du LLM


@app.on_event("startup")
def warm_up():
    get_embedder()          # charge le modèle une fois, pas à la première requête
    _client()
    try:
        get_index()         # construit l'index BM25 en mémoire
    except FileNotFoundError:
        pass                # chunks.jsonl absent : /search en mode hybrid retombera sur dense


@app.get("/health")
def health():
    try:
        n = _client().count(config.COLLECTION, exact=True).count
    except Exception as e:
        raise HTTPException(503, f"Qdrant injoignable ou collection absente : {type(e).__name__}")
    try:
        bm25_chunks = get_index().n
    except FileNotFoundError:
        bm25_chunks = None
    return {"status": "ok", "collection": config.COLLECTION, "points": n,
            "bm25_chunks": bm25_chunks, "bm25_available": bm25_chunks is not None}


@app.post("/search", response_model=SearchResponse)
def search_endpoint(req: SearchRequest):
    try:
        hits = search(req.query, k=req.k, source=req.source, min_score=req.min_score, mode=req.mode)
    except Exception as e:
        raise HTTPException(503, f"Recherche impossible : {type(e).__name__}")
    return SearchResponse(mode=req.mode, hits=hits, context=format_context(hits))