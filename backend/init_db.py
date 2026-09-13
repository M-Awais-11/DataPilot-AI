"""Database schema initialization for Microsoft SQL Server."""
import urllib.parse
from sqlalchemy import create_engine
from app.config.settings import settings
from app.database.connection import Base
import app.models  # Register all models with Base.metadata

def init_tables():
    sync_url = settings.get_sync_db_url()
    print(f"Connecting to SQL Server: {settings.sql_server} / {settings.sql_database}")
    engine = create_engine(sync_url, echo=False)
    
    print("Creating all tables in SQL Server...")
    Base.metadata.create_all(engine)
    print("[OK] Successfully initialized all tables in Microsoft SQL Server!")

if __name__ == "__main__":
    init_tables()
