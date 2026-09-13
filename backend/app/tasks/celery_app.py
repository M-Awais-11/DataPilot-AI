"""Celery entrypoint module."""
from app.tasks.experiment_tasks import celery_app

__all__ = ["celery_app"]
