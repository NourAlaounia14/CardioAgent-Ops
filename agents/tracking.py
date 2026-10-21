"""Suivi des guardrails sous MLflow (Service 6). Désactivé par défaut."""
import logging
import os

logger = logging.getLogger("agent-orchestrator")


def log_run(final: dict) -> None:
    if os.getenv("MLFLOW_ENABLED", "false").lower() != "true":
        return
    try:
        import mlflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000"))
        mlflow.set_experiment("agent-guardrails")
        with mlflow.start_run(run_name=final["request_id"]):
            mlflow.log_metrics({"attempts": final["retries"], "audit_ok": int(final["audit_ok"]),
                                "p_abnormal": final["p_abnormal"]})
            mlflow.set_tag("risk_level", final["report"]["risk_level"])
    except Exception:   # le suivi ne doit jamais casser une requête médicale
        logger.warning("Échec du log MLflow", exc_info=True)
