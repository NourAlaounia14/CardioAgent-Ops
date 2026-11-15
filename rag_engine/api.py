"""rag_engine/api.py : API FastAPI du RAG (utilisable seule, ou appelée par le KnowledgeRAG-Agent).

Lancement :   uvicorn rag_engine.api:app --host 0.0.0.0 --port 8001
Test :        curl -X POST http://localhost:8001/search -H "Content-Type: application/json" \
                   -d '{"query": "souffle systolique, conduite à tenir", "k": 3}'
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_engine import config
from rag_engine.embedder import get_embedder
from rag_engine.search import _client, format_context, search

app = FastAPI(title="rag-engine")


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    k: int = Field(4, ge=1, le=10)
    source: str | None = None
    min_score: float = Field(0.0, ge=0.0, le=1.0)


class Hit(BaseModel):
    score: float
    text: str
    heading: str
    source: str
    page: int
    chunk_id: str


class SearchResponse(BaseModel):
    hits: list[Hit]
    context: str            # bloc prêt à insérer dans le prompt du LLM


@app.on_event("startup")
def warm_up():
    get_embedder()          # charge le modèle une fois, pas à la première requête
    _client()


@app.get("/health")
def health():
    try:
        n = _client().count(config.COLLECTION, exact=True).count
    except Exception as e:
        raise HTTPException(503, f"Qdrant injoignable ou collection absente : {type(e).__name__}")
    return {"status": "ok", "collection": config.COLLECTION, "points": n}


@app.post("/search", response_model=SearchResponse)
def search_endpoint(req: SearchRequest):
    try:
        hits = search(req.query, k=req.k, source=req.source, min_score=req.min_score)
    except Exception as e:
        raise HTTPException(503, f"Recherche impossible : {type(e).__name__}")
    return SearchResponse(hits=hits, context=format_context(hits))
