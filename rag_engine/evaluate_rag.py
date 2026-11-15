"""rag_engine/evaluate_rag.py : mesure la qualité de la recherche (section « évaluation du RAG » du rapport).

Pour chaque question, on vérifie si un des k premiers passages contient la réponse, repérée par des
mots-clés que TU choisis en lisant le PDF.

Format de rag_engine/eval_questions.json :
[
  {"question": "Quand recommander une échocardiographie devant un souffle systolique ?",
   "expected_keywords": ["echocardiograph", "murmur"],
   "match": "all",                       # "all" (défaut) : tous les mots-clés, "any" : au moins un
   "expected_source": "esc_valvular.pdf"}   # optionnel
]

Métriques : Recall@1/3/5 (la bonne réponse est-elle dans le top k ?) et MRR (rang moyen inversé).
Usage :  python -m rag_engine.evaluate_rag --questions rag_engine/eval_questions.json [--mlflow]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from rag_engine import config
from rag_engine.search import search

KS = (1, 3, 5)


def first_hit_rank(q: dict, hits: list[dict]) -> int | None:
    kws = [k.lower() for k in q["expected_keywords"]]
    agg = all if q.get("match", "all") == "all" else any
    for rank, h in enumerate(hits, start=1):
        if q.get("expected_source") and h["source"] != q["expected_source"]:
            continue
        text = (h["heading"] + " " + h["text"]).lower()
        if agg(k in text for k in kws):
            return rank
    return None


def evaluate(questions: list[dict]):
    rows = []
    for q in questions:
        hits = search(q["question"], k=max(KS))
        rank = first_hit_rank(q, hits)
        rows.append(dict(question=q["question"], rank=rank, top1_score=hits[0]["score"] if hits else None,
                         top1_source=hits[0]["source"] if hits else None,
                         top1_page=hits[0]["page"] if hits else None))
    df = pd.DataFrame(rows)
    metrics = {f"recall_at_{k}": float(((df["rank"].notna()) & (df["rank"] <= k)).mean()) for k in KS}
    metrics["mrr"] = float((1 / df["rank"]).fillna(0).mean())
    return df, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(config.ROOT / "eval_questions.json"))
    ap.add_argument("--out", default=str(config.ROOT / "data" / "rag_eval_results.csv"))
    ap.add_argument("--mlflow", action="store_true")
    a = ap.parse_args()

    questions = json.loads(Path(a.questions).read_text(encoding="utf-8"))
    df, m = evaluate(questions)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df.to_string(index=False))
    print("\n" + " | ".join(f"{k} = {v:.3f}" for k, v in m.items()), f"| {len(df)} questions")

    if a.mlflow:
        import mlflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_URI", "http://localhost:5000"))
        mlflow.set_experiment("rag-evaluation")
        with mlflow.start_run(run_name=f"rag_{config.EMBEDDER}_{config.EMBED_MODEL.split('/')[-1]}"):
            mlflow.log_params(dict(embedder=config.EMBEDDER, embed_model=config.EMBED_MODEL,
                                   max_words=config.MAX_WORDS, overlap=config.OVERLAP_WORDS,
                                   n_questions=len(df)))
            mlflow.log_metrics(m)
            mlflow.log_artifact(a.out)


if __name__ == "__main__":
    main()
