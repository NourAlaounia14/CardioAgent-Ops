"""rag_engine/search.py : recherche utilisée par le KnowledgeRAG-Agent.

    from rag_engine.search import search, format_context
    hits = search("souffle systolique anormal, conduite à tenir", k=4)          # hybride (défaut)
    hits = search("ETT", k=4, mode="dense")                                      # embeddings seuls
    contexte = format_context(hits)        # texte à insérer dans le prompt du LLM

Modes : "hybrid" (dense + BM25, fusion RRF) | "dense" (e5 + Qdrant) | "bm25" (mots-clés).

CLI :  python -m rag_engine.search "aortic stenosis echocardiography" -k 3 --mode hybrid
"""
from __future__ import annotations

import argparse
import warnings
from functools import lru_cache

from qdrant_client.models import FieldCondition, Filter, MatchValue

from rag_engine import config
from rag_engine.Bm25 import bm25_search
from rag_engine.embed_index import get_client
from rag_engine.embedder import get_embedder

RRF_K = 60            # constante standard de la fusion RRF
CANDIDATES = 20       # nombre de candidats pris dans chaque liste avant fusion


@lru_cache(maxsize=1)
def _client():
    return get_client()


def _dense(query: str, k: int, source: str | None) -> list[dict]:
    flt = Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))]) if source else None
    res = _client().query_points(config.COLLECTION, query=get_embedder().embed_query(query).tolist(),
                                 limit=k, query_filter=flt, with_payload=True)
    return [dict(score=round(float(p.score), 4), text=p.payload["text"], heading=p.payload.get("heading", ""),
                 source=p.payload["source"], page=p.payload["page_start"], chunk_id=p.payload["chunk_id"])
            for p in res.points]


def rrf_fuse(dense: list[dict], lexical: list[dict], k: int, rrf_k: int = RRF_K) -> list[dict]:
    """Reciprocal Rank Fusion : score = somme de 1/(rrf_k + rang) sur les deux listes."""
    fused: dict[str, dict] = {}
    for name, hits in (("dense", dense), ("bm25", lexical)):
        for rank, h in enumerate(hits, start=1):
            e = fused.setdefault(h["chunk_id"], dict(h, score=0.0, dense_score=None, bm25_score=None,
                                                     dense_rank=None, bm25_rank=None))
            e["score"] += 1.0 / (rrf_k + rank)
            e[f"{name}_score"] = h["score"]
            e[f"{name}_rank"] = rank
    out = sorted(fused.values(), key=lambda h: -h["score"])[:k]
    for h in out:
        h["score"] = round(h["score"], 4)
    return out


def search(query: str, k: int = 4, source: str | None = None, min_score: float = 0.0,
           mode: str = "hybrid") -> list[dict]:
    """Renvoie les k passages les plus pertinents, du plus au moins pertinent.

    min_score ne s'applique qu'au mode "dense" (similarité cosinus) ; en "hybrid" le score est un
    score de fusion RRF (petit, ~0.01-0.03) et chaque hit garde dense_score / bm25_score / rangs.
    """
    if mode == "dense":
        return [h for h in _dense(query, k, source) if h["score"] >= min_score]
    if mode == "bm25":
        return bm25_search(query, k, source)
    if mode != "hybrid":
        raise ValueError("mode doit être 'hybrid', 'dense' ou 'bm25'")
    n = max(CANDIDATES, k * 3)
    dense = _dense(query, n, source)
    try:
        lexical = bm25_search(query, n, source)
    except FileNotFoundError as e:             # pas de chunks.jsonl (ex. API déployée sans) : repli
        warnings.warn(f"BM25 indisponible, repli sur dense : {e}")
        return dense[:k]
    return rrf_fuse(dense, lexical, k)


def format_context(hits: list[dict]) -> str:
    """Bloc de contexte numéroté, avec source et page pour que le LLM puisse les citer."""
    return "\n\n".join(f"[{i}] ({h['source']}, p.{h['page']}"
                       f"{', ' + h['heading'] if h['heading'] else ''})\n{h['text']}"
                       for i, h in enumerate(hits, start=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--source", default=None)
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "bm25"])
    a = ap.parse_args()
    for h in search(a.query, a.k, a.source, mode=a.mode):
        extra = (f" (dense #{h['dense_rank']}, bm25 #{h['bm25_rank']})" if a.mode == "hybrid" else "")
        print(f"\n[{h['score']}]{extra} {h['source']} p.{h['page']} | {h['heading']}\n{h['text'][:400]}...")