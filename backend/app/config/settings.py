"""
Application configuration using Pydantic Settings.
Configured for Microsoft SQL Server and OpenRouter.
"""
from functools import lru_cache
from typing import List, Optional
import urllib.parse
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ─────────────────────────────────────────────────────────
    app_name: str = "AI Autonomous Data Scientist Agent"
    app_version: str = "1.0.0"
    debug: bool = True
    allowed_origins: List[str] = ["http://localhost:4200", "http://localhost:3000", "http://127.0.0.1:4200"]

    # ── Microsoft SQL Server Connection ──────────────────────────────────────
    sql_server: str = "AWAIS\\SQLEXPRESS"
    sql_database: str = "AI_Autonomous_Data_Scientist_Agent"
    sql_driver: str = "ODBC Driver 18 for SQL Server"
    sql_trusted_connection: str = "yes"
    sql_trust_server_certificate: str = "yes"

    # Explicit full URLs if overridden
    database_url: Optional[str] = None
    sync_database_url: Optional[str] = None

    # ── Security ─────────────────────────────────────────────────────────────
    secret_key: str = "openrouter-autonomous-data-scientist-secret-key-2026"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 120
    refresh_token_expire_days: int = 7

    # ── OpenRouter LLM ───────────────────────────────────────────────────────
    openrouter_api_key: str = "YOUR_REAL_KEY"
    openrouter_model: str = "openai/gpt-4o-mini"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_temperature: float = 0.1

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = "./mlruns"

    # ── Storage ───────────────────────────────────────────────────────────────
    upload_dir: str = "./uploads"
    reports_dir: str = "./reports"
    max_upload_size_mb: int = 100

    # ── ML Defaults ───────────────────────────────────────────────────────────
    random_seed: int = 42
    cv_folds: int = 5
    test_size: float = 0.2
    validation_size: float = 0.1
    optuna_trials: int = 20
    max_training_time_seconds: int = 3600

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_origins(cls, v):
        if isinstance(v, str):
            return [o.strip() for o in v.split(",")]
        return v

    def get_async_db_url(self) -> str:
        if self.database_url:
            return self.database_url
        params = urllib.parse.quote_plus(
            f"DRIVER={{{self.sql_driver}}};"
            f"SERVER={self.sql_server};"
            f"DATABASE={self.sql_database};"
            f"Trusted_Connection={self.sql_trusted_connection};"
            f"TrustServerCertificate={self.sql_trust_server_certificate};"
        )
        return f"mssql+aioodbc:///?odbc_connect={params}"

    def get_sync_db_url(self) -> str:
        if self.sync_database_url:
            return self.sync_database_url
        params = urllib.parse.quote_plus(
            f"DRIVER={{{self.sql_driver}}};"
            f"SERVER={self.sql_server};"
            f"DATABASE={self.sql_database};"
            f"Trusted_Connection={self.sql_trusted_connection};"
            f"TrustServerCertificate={self.sql_trust_server_certificate};"
        )
        return f"mssql+pyodbc:///?odbc_connect={params}"

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
