"""rag_engine/search.py : recherche utilisée par le KnowledgeRAG-Agent.

    from rag_engine.search import search, format_context
    hits = search("souffle systolique anormal, conduite à tenir", k=4)
    contexte = format_context(hits)        # texte à insérer dans le prompt du LLM

CLI :  python -m rag_engine.search "aortic stenosis echocardiography" -k 3
"""
from __future__ import annotations

import argparse
from functools import lru_cache

from qdrant_client.models import FieldCondition, Filter, MatchValue

from rag_engine import config
from rag_engine.embed_index import get_client
from rag_engine.embedder import get_embedder


@lru_cache(maxsize=1)
def _client():
    return get_client()


def search(query: str, k: int = 4, source: str | None = None, min_score: float = 0.0) -> list[dict]:
    """Renvoie les k passages les plus proches, du plus au moins pertinent."""
    flt = Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))]) if source else None
    res = _client().query_points(config.COLLECTION, query=get_embedder().embed_query(query).tolist(),
                                 limit=k, query_filter=flt, with_payload=True)
    return [dict(score=round(float(p.score), 4), text=p.payload["text"], heading=p.payload.get("heading", ""),
                 source=p.payload["source"], page=p.payload["page_start"], chunk_id=p.payload["chunk_id"])
            for p in res.points if p.score >= min_score]


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
    a = ap.parse_args()
    for h in search(a.query, a.k, a.source):
        print(f"\n[{h['score']}] {h['source']} p.{h['page']} | {h['heading']}\n{h['text'][:400]}...")
