"""Tests du Service 4 : schéma de sortie, graphe, guardrails de l'Auditeur."""
import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["USE_MOCKS"] = "true"

import clients  # noqa: E402
from main import app  # noqa: E402
from schemas import AnalysisReport  # noqa: E402

client = TestClient(app)


def payload(p_abnormal: float) -> dict:
    cls = "anormal" if p_abnormal >= 0.5 else "normal"
    return {"request_id": "test-1", "audio_features": {
        "predicted_class": cls, "confidence": max(p_abnormal, 1 - p_abnormal),
        "probabilities": {"normal": round(1 - p_abnormal, 4), "anormal": p_abnormal}}}


def llm_returning(*answers):
    """Faux LLM : renvoie les réponses dans l'ordre et compte les appels."""
    calls = {"n": 0}

    def fake(messages, mock_context=None):
        ans = answers[min(calls["n"], len(answers) - 1)]
        calls["n"] += 1
        return ans if isinstance(ans, str) else json.dumps(ans)
    return fake, calls


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_abnormal_signal_gives_high_risk_with_recommendation():
    r = client.post("/orchestrate", json=payload(0.9))
    assert r.status_code == 200
    report = AnalysisReport.model_validate(r.json())      # conformité du schéma
    assert report.risk_level == "high"
    assert len(report.clinical_recommendations) >= 1
    assert "Validé" in report.audit_notes


def test_normal_signal_gives_low_risk():
    r = client.post("/orchestrate", json=payload(0.1))
    assert r.status_code == 200
    assert r.json()["risk_level"] == "low"
    assert r.json()["clinical_recommendations"] == []


def test_invalid_request_rejected():
    bad = payload(0.9)
    bad["audio_features"]["confidence"] = 1.7
    assert client.post("/orchestrate", json=bad).status_code == 422


def test_auditor_rejects_underestimated_risk_then_falls_back(monkeypatch):
    bad = {"diagnosis": "Rien d'inquiétant détecté ici.", "risk_level": "low", "cited_sources": []}
    fake, calls = llm_returning(bad)
    monkeypatch.setattr(clients, "call_llm", fake)
    r = client.post("/orchestrate", json=payload(0.9))
    assert r.status_code == 200
    assert calls["n"] == 2                                  # 2 tentatives puis repli
    assert r.json()["risk_level"] == "high"                 # jamais "low" sur signal anormal
    assert "REPLI" in r.json()["audit_notes"]


def test_auditor_rejects_hallucinated_source_then_accepts_correction(monkeypatch):
    hallucinated = {"diagnosis": "Anomalie probable du signal.", "risk_level": "high",
                    "cited_sources": ["Source inventée 2099"]}
    good = {"diagnosis": "Anomalie probable du signal.", "risk_level": "high",
            "cited_sources": ["ESC Guidelines 2021 - Valvular Heart Disease"]}
    fake, calls = llm_returning(hallucinated, good)
    monkeypatch.setattr(clients, "call_llm", fake)
    r = client.post("/orchestrate", json=payload(0.9))
    assert calls["n"] == 2
    assert "tentative 2/2" in r.json()["audit_notes"]
    assert r.json()["clinical_recommendations"][0]["source"] == "ESC Guidelines 2021 - Valvular Heart Disease"


def test_auditor_rejects_non_json_llm_output(monkeypatch):
    fake, calls = llm_returning("ceci n'est pas du json")
    monkeypatch.setattr(clients, "call_llm", fake)
    r = client.post("/orchestrate", json=payload(0.9))
    assert r.status_code == 200
    assert calls["n"] == 2
    assert "REPLI" in r.json()["audit_notes"]
