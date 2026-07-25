"""
FastAPI application entrypoint.

Startup tasks:
  1. Enable the pgvector extension in Postgres.
  2. Create the `predictions` and `policy_chunks` tables if they don't exist.

Routes:
  GET  /                  → health check
  POST /api/predict-loan  → ML inference
  GET  /api/predictions   → history
  GET  /api/feature-importance → model coefficients
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text

from app.routers import prediction, agent as agent_router
from app.utils.db import engine, enable_pgvector


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── startup ──────────────────────────────────────────────────────────────
    enable_pgvector()

    with engine.connect() as conn:
        # Loan prediction history
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id           SERIAL PRIMARY KEY,
                    age          INT,
                    salary       FLOAT,
                    credit_score INT,
                    loan_amount  FLOAT,
                    loan_status  TEXT,
                    probability  FLOAT,
                    created_at   TIMESTAMP DEFAULT now()
                )
                """
            )
        )

        # RAG vector store for policy documents
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS policy_chunks (
                    id        SERIAL PRIMARY KEY,
                    content   TEXT,
                    embedding VECTOR(384),
                    source    TEXT
                )
                """
            )
        )

        # Speed up cosine-distance queries with an IVFFlat index (created once)
        conn.execute(
            text(
                """
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_indexes
                        WHERE tablename = 'policy_chunks'
                          AND indexname = 'policy_chunks_embedding_idx'
                    ) THEN
                        CREATE INDEX policy_chunks_embedding_idx
                        ON policy_chunks
                        USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 10);
                    END IF;
                END $$;
                """
            )
        )

        conn.commit()

    print("✓ Database tables ready.")
    yield
    # ── shutdown ─────────────────────────────────────────────────────────────
    # Nothing to clean up — connection pool handles itself.


app = FastAPI(
    title="Loan Prediction API",
    description=(
        "ML-powered loan approval prediction with a RAG pipeline over underwriting "
        "policy documents, backed by PostgreSQL + pgvector."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(prediction.router, prefix="/api")
app.include_router(agent_router.router, prefix="/api")


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Loan Prediction API is running"}
