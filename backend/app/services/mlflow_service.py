"""
MLflow Tracking and Model Registry Service.
Logs parameters, metrics, artifacts, and registers champion models.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Any
import mlflow
from mlflow.tracking import MlflowClient

from app.config.settings import settings

logger = logging.getLogger(__name__)


class MLflowService:
    def __init__(self, tracking_uri: Optional[str] = None):
        self.tracking_uri = tracking_uri or settings.mlflow_tracking_uri
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            self.client = MlflowClient(tracking_uri=self.tracking_uri)
            self.enabled = True
        except Exception as e:
            logger.warning(f"MLflow service initialization failed or server unreachable ({e}). Logging locally.")
            self.enabled = False

    def get_or_create_experiment(self, experiment_name: str) -> str:
        """Return experiment ID, creating it if not present."""
        if not self.enabled:
            return "local_exp"
        try:
            exp = mlflow.get_experiment_by_name(experiment_name)
            if exp:
                return exp.experiment_id
            return mlflow.create_experiment(experiment_name)
        except Exception as e:
            logger.warning(f"MLflow get_or_create_experiment error: {e}")
            return "default"

    def log_run(
        self,
        experiment_id: str,
        run_name: str,
        parameters: dict[str, Any],
        metrics: dict[str, float],
        tags: Optional[dict[str, str]] = None,
        artifact_paths: Optional[list[str]] = None,
    ) -> Optional[str]:
        """Log a complete training run to MLflow."""
        if not self.enabled:
            return None

        try:
            with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
                # Log tags
                if tags:
                    mlflow.set_tags(tags)
                
                # Log parameters (flatten nested if needed)
                flat_params = {k: str(v)[:250] for k, v in parameters.items()}
                mlflow.log_params(flat_params)

                # Log metrics
                for m_name, m_val in metrics.items():
                    if isinstance(m_val, (int, float)):
                        mlflow.log_metric(m_name, float(m_val))

                # Log artifacts
                if artifact_paths:
                    for art in artifact_paths:
                        if os.path.exists(art):
                            mlflow.log_artifact(art)

                return run.info.run_id
        except Exception as e:
            logger.warning(f"Failed to log run to MLflow: {e}")
            return None

    def register_production_model(
        self,
        model_name: str,
        model_uri: str,
        version_description: str = "Promoted by Autonomous Data Scientist Agent",
    ) -> Optional[str]:
        """Register model version and transition to Production."""
        if not self.enabled:
            return None

        try:
            reg_model = mlflow.register_model(model_uri=model_uri, name=model_name)
            self.client.transition_model_version_stage(
                name=model_name,
                version=reg_model.version,
                stage="Production",
                archive_existing_versions=True,
            )
            self.client.update_model_version(
                name=model_name,
                version=reg_model.version,
                description=version_description,
            )
            logger.info(f"Registered {model_name} version {reg_model.version} in Production stage")
            return reg_model.version
        except Exception as e:
            logger.warning(f"Failed to register production model in MLflow: {e}")
            return None
