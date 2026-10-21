"""Graphe LangGraph à 3 agents : Analyste -> Cardiologue (RAG + LLM) -> Auditeur.

L'Auditeur applique les guardrails ; s'il rejette, le Cardiologue réessaie
(MAX_ATTEMPTS essais au total), sinon un rapport de repli prudent est renvoyé.
"""
import os
from datetime import datetime, timezone
from typing import TypedDict

from langgraph.graph import END, StateGraph
from pydantic import ValidationError

import clients
from prompts import SYSTEM_PROMPT, build_user_prompt
from schemas import AnalysisReport, AudioFeatures, ClinicalRecommendation, DraftReport

RISK_ORDER = ["low", "moderate", "high", "critical"]
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "2"))
P_ABNORMAL = float(os.getenv("P_ABNORMAL_THRESHOLD", "0.5"))
P_HIGH = float(os.getenv("P_ABNORMAL_HIGH", "0.8"))


class OrchestratorState(TypedDict, total=False):
    request_id: str
    audio: dict
    p_abnormal: float
    summary: str
    query: str
    passages: list[dict]
    raw_llm: str
    draft: dict | None
    parse_error: str | None
    feedback: str | None
    retries: int
    audit_ok: bool
    report: dict


# ------------------------------------------------------------------ utilitaires
def abnormal_probability(audio: dict) -> float:
    probs = audio.get("probabilities") or {}
    for key in ("anormal", "abnormal", "pathologique"):
        if key in probs:
            return float(probs[key])
    conf = float(audio["confidence"])
    return conf if audio["predicted_class"] in ("anormal", "abnormal") else 1 - conf


def min_risk(p_abnormal: float) -> str:
    """Niveau de risque MINIMAL cohérent avec la probabilité d'anomalie."""
    if p_abnormal >= P_HIGH:
        return "high"
    if p_abnormal >= P_ABNORMAL:
        return "moderate"
    return "low"


def _rank(risk: str) -> int:
    return RISK_ORDER.index(risk)


# ------------------------------------------------------------- agent 1 : Analyste
def analyste(state: OrchestratorState) -> dict:
    audio = state["audio"]
    p = abnormal_probability(audio)
    summary = (f"Classe prédite : {audio['predicted_class']} "
               f"(confiance {audio['confidence']:.0%}). Probabilité d'anomalie : {p:.0%}.")
    query = ("recommandations ESC conduite à tenir souffle cardiaque anormal"
             if p >= P_ABNORMAL else "recommandations ESC auscultation normale suivi")
    return {"p_abnormal": p, "summary": summary, "query": query, "retries": 0}


# --------------------------------------------------- agent 2 : Cardiologue (RAG)
def cardiologue(state: OrchestratorState) -> dict:
    passages = state.get("passages") or clients.retrieve_guidelines(state["query"])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(state["summary"], passages, state.get("feedback"))},
    ]
    raw = clients.call_llm(messages, mock_context={
        "p_abnormal": state["p_abnormal"], "sources": [p["source"] for p in passages]})
    try:
        draft, error = DraftReport.model_validate_json(raw).model_dump(), None
    except ValidationError as e:
        draft = None
        error = "; ".join(f"{'.'.join(map(str, err['loc'])) or 'json'}: {err['msg']}" for err in e.errors())
    return {"passages": passages, "raw_llm": raw, "draft": draft, "parse_error": error}


# ------------------------------------------------------------ agent 3 : Auditeur
def check_draft(state: OrchestratorState) -> list[str]:
    """Guardrails : liste des problèmes détectés (vide = brouillon accepté)."""
    draft = state.get("draft")
    if draft is None:
        return [f"sortie du LLM non conforme au schéma JSON ({state.get('parse_error')})"]
    problems = []
    p = state["p_abnormal"]
    if _rank(draft["risk_level"]) < _rank(min_risk(p)):
        problems.append(f"risque '{draft['risk_level']}' sous-estimé (probabilité d'anomalie {p:.0%})")
    if p < P_ABNORMAL and _rank(draft["risk_level"]) >= _rank("high"):
        problems.append(f"risque '{draft['risk_level']}' injustifié (probabilité d'anomalie {p:.0%})")
    known = {x["source"] for x in state["passages"]}
    unknown = [s for s in draft["cited_sources"] if s not in known]
    if unknown:
        problems.append(f"sources absentes du contexte RAG (hallucination) : {unknown}")
    if p >= P_ABNORMAL and not draft["cited_sources"]:
        problems.append("aucune recommandation ESC citée pour un signal anormal")
    return problems


def _build_report(state: OrchestratorState, diagnosis: str, risk: str,
                  recos: list[dict], notes: str) -> dict:
    audio = AudioFeatures(**state["audio"])
    report = AnalysisReport(
        request_id=state["request_id"], diagnosis=diagnosis, confidence=audio.confidence,
        risk_level=risk, audio_features=audio,
        clinical_recommendations=[ClinicalRecommendation(**r) for r in recos],
        audit_notes=notes, timestamp=datetime.now(timezone.utc))
    return report.model_dump(mode="json")


def auditeur(state: OrchestratorState) -> dict:
    problems = check_draft(state)
    attempts = state.get("retries", 0) + 1
    if not problems:
        draft = state["draft"]
        recos = [x for x in state["passages"] if x["source"] in draft["cited_sources"]]
        notes = f"Validé par l'Auditeur (tentative {attempts}/{MAX_ATTEMPTS})."
        return {"audit_ok": True, "retries": attempts,
                "report": _build_report(state, draft["diagnosis"], draft["risk_level"], recos, notes)}
    if attempts >= MAX_ATTEMPTS:
        p = state["p_abnormal"]
        risk = "moderate" if (p >= P_ABNORMAL and min_risk(p) == "low") else min_risk(p)
        diagnosis = ("Signal compatible avec une anomalie cardiaque. Rapport automatique non validé : avis cardiologique requis."
                     if p >= P_ABNORMAL else
                     "Aucune anomalie détectée par l'analyse automatique. Rapport non validé : confirmer par un avis médical.")
        notes = f"RAPPORT DE REPLI après {attempts} tentatives. Problèmes : {' | '.join(problems)}"
        return {"audit_ok": False, "retries": attempts,
                "report": _build_report(state, diagnosis, risk, [], notes)}
    return {"audit_ok": False, "retries": attempts, "feedback": "; ".join(problems)}


# ----------------------------------------------------------------------- graphe
def build_graph():
    g = StateGraph(OrchestratorState)
    g.add_node("analyste", analyste)
    g.add_node("cardiologue", cardiologue)
    g.add_node("auditeur", auditeur)
    g.set_entry_point("analyste")
    g.add_edge("analyste", "cardiologue")
    g.add_edge("cardiologue", "auditeur")
    g.add_conditional_edges("auditeur",
                            lambda s: "done" if s.get("report") else "retry",
                            {"done": END, "retry": "cardiologue"})
    return g.compile()


GRAPH = build_graph()


def run(request_id: str, audio: dict) -> dict:
    """Exécute le graphe ; renvoie l'état final (clé 'report' = rapport validé)."""
    return GRAPH.invoke({"request_id": request_id, "audio": audio})
