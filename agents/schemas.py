"""Schémas Pydantic du Service 4 (agent-orchestrator).

AnalysisReport est IDENTIQUE à celui de l'api-gateway : c'est le contrat de sortie.
DraftReport est ce que le LLM doit produire (validé par l'Auditeur = guardrail).
"""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

RiskLevel = Literal["low", "moderate", "high", "critical"]


class AudioFeatures(BaseModel):
    predicted_class: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]


class OrchestrateRequest(BaseModel):
    request_id: str
    audio_features: AudioFeatures


class ClinicalRecommendation(BaseModel):
    source: str
    excerpt: str
    similarity_score: float


class DraftReport(BaseModel):
    """Sortie attendue du LLM (Agent Cardiologue)."""
    diagnosis: str = Field(min_length=10)
    risk_level: RiskLevel
    cited_sources: list[str] = []


class AnalysisReport(BaseModel):
    request_id: str
    diagnosis: str
    confidence: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    audio_features: AudioFeatures
    clinical_recommendations: list[ClinicalRecommendation] = []
    audit_notes: str | None = None
    timestamp: datetime
