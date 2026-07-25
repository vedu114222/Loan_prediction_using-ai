import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

_PG_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/loans",
)

# ── Auto-detect: fall back to SQLite if Postgres is unavailable ───────────────
def _make_engine():
    """Try PostgreSQL first; silently fall back to SQLite for local dev."""
    if "sqlite" in _PG_URL:
        # Already explicitly set to SQLite
        return create_engine(_PG_URL, connect_args={"check_same_thread": False})
    try:
        eng = create_engine(_PG_URL, pool_pre_ping=True)
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("[OK] Connected to PostgreSQL.")
        return eng
    except Exception:
        sqlite_url = "sqlite:///./loan_local.db"
        print("WARNING: PostgreSQL unavailable -- falling back to SQLite for local testing.")
        print(f"   Database file: loan_local.db")

        return create_engine(sqlite_url, connect_args={"check_same_thread": False})


engine = _make_engine()
DATABASE_URL = str(engine.url)   # reflect actual URL (pg or sqlite)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

IS_SQLITE = engine.dialect.name == "sqlite"


def enable_pgvector() -> None:
    """Enable pgvector extension (no-op when using SQLite)."""
    if IS_SQLITE:
        return
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

