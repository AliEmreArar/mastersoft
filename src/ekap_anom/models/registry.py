"""Model registry: MLflow integration for save/load/promote."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ekap_anom.logging import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """MLflow-based model registry for training artifacts.

    Falls back to local file system if MLflow is unavailable.
    """

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "ekap-anomaly",
        registry_name: str = "ekap-anom-models",
    ) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.registry_name = registry_name
        self._mlflow: Any = None

    def _get_mlflow(self) -> Any:
        """Lazily import and configure MLflow."""
        if self._mlflow is None:
            try:
                import mlflow
                mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(self.experiment_name)
                self._mlflow = mlflow
                logger.info("mlflow_connected", uri=self.tracking_uri)
            except Exception as e:
                logger.warning("mlflow_unavailable", error=str(e))
                self._mlflow = None
        return self._mlflow

    def log_model(
        self,
        model_path: str | Path,
        model_name: str,
        metrics: dict[str, float] | None = None,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
    ) -> str | None:
        """Log a model artifact to MLflow.

        Args:
            model_path: Local path to serialized model.
            model_name: Name for the model artifact.
            metrics: Training metrics to log.
            params: Hyperparameters to log.
            tags: Extra tags.

        Returns:
            Run ID or None if MLflow is unavailable.
        """
        mlflow = self._get_mlflow()
        if mlflow is None:
            logger.info("model_logged_locally", path=str(model_path), name=model_name)
            return None

        with mlflow.start_run(run_name=model_name) as run:
            if params:
                mlflow.log_params(params)
            if metrics:
                mlflow.log_metrics(metrics)
            if tags:
                mlflow.set_tags(tags)

            mlflow.log_artifact(str(model_path))
            run_id = run.info.run_id
            logger.info("model_logged_mlflow", run_id=run_id, name=model_name)
            return run_id

    def register_model(
        self,
        run_id: str,
        model_name: str,
    ) -> str | None:
        """Register a logged model in the MLflow registry.

        Args:
            run_id: MLflow run ID.
            model_name: Registry model name.

        Returns:
            Model version string or None.
        """
        mlflow = self._get_mlflow()
        if mlflow is None:
            return None

        try:
            result = mlflow.register_model(
                f"runs:/{run_id}/model",
                model_name,
            )
            version = result.version
            logger.info("model_registered", name=model_name, version=version)
            return str(version)
        except Exception as e:
            logger.warning("model_registration_failed", error=str(e))
            return None

    def load_latest(self, model_name: str) -> Path | None:
        """Load the latest model artifact from MLflow.

        Returns local path to the artifact, or None if unavailable.
        """
        mlflow = self._get_mlflow()
        if mlflow is None:
            return None

        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            versions = client.get_latest_versions(model_name)
            if not versions:
                return None
            latest = versions[0]
            artifact_uri = latest.source
            local_path = mlflow.artifacts.download_artifacts(artifact_uri)
            return Path(local_path)
        except Exception as e:
            logger.warning("model_load_failed", error=str(e))
            return None
