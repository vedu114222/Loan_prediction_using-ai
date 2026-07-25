"""
MCP server — exposes loan prediction, policy search, prediction history, and
feature importance as LLM-callable tools via the Model Context Protocol.

Start it directly:
    python -m app.mcp.server

Or register it in claude_desktop_config.json:
    {
      "mcpServers": {
        "loan-prediction": {
          "command": "C:/path/to/venv/Scripts/python.exe",
          "args": ["-m", "app.mcp.server"],
          "cwd": "C:/path/to/loan-prediction-api"
        }
      }
    }
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Ensure the project root is on sys.path when run as __main__
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from app.rag.retriever import search_policy_chunks  # noqa: E402
from app.services.prediction_service import PredictionService  # noqa: E402
from app.utils.db import SessionLocal  # noqa: E402
from sqlalchemy import text  # noqa: E402

mcp = FastMCP(
    "loan-prediction",
    instructions=(
        "You are a loan underwriting assistant with access to four tools:\n"
        "1. predict_loan      — run the ML model on applicant data\n"
        "2. search_policy     — semantic search over underwriting policy documents\n"
        "3. get_history       — fetch recent predictions from the database\n"
        "4. feature_importance — see which applicant factors the model weights most\n\n"
        "Always call search_policy to ground your explanations in actual bank policy "
        "before drawing conclusions. Cite the source document in your answers."
    ),
)


# ── Tool 1: ML inference ──────────────────────────────────────────────────────

@mcp.tool()
def predict_loan(
    age: int,
    salary: float,
    credit_score: int,
    loan_amount: float,
) -> dict:
    """
    Run the logistic regression model on an applicant's data and return a
    loan decision with a probability score.

    Args:
        age:          Applicant age in years (18–100).
        salary:       Annual gross salary in USD.
        credit_score: FICO credit score (300–850).
        loan_amount:  Requested loan amount in USD.

    Returns:
        dict with keys: loan_status ("Approved" | "Rejected"), probability (0–1),
        approved (bool).
    """
    return PredictionService.predict_raw(age, salary, credit_score, loan_amount)


# ── Tool 2: Policy RAG search ─────────────────────────────────────────────────

@mcp.tool()
def search_policy(query: str, top_k: int = 4) -> list[dict]:
    """
    Semantic search over the bank's underwriting policy documents stored in
    pgvector. Use this to ground any loan decision explanation in actual policy
    language rather than general knowledge.

    Args:
        query: Natural-language question, e.g. "minimum credit score for approval"
               or "DTI limit for conventional mortgage".
        top_k: Number of policy passages to return (default 4, max 10).

    Returns:
        List of dicts with keys: content (policy text), source (filename), score (0–1).
        Higher score = more semantically relevant.
    """
    top_k = min(top_k, 10)
    return search_policy_chunks(query, top_k)


# ── Tool 3: Prediction history ────────────────────────────────────────────────

@mcp.tool()
def get_history(limit: int = 10) -> list[dict]:
    """
    Retrieve the most recent loan predictions stored in PostgreSQL.
    Useful for spotting patterns across recent applications.

    Args:
        limit: Number of records to return (default 10, max 100).

    Returns:
        List of dicts with: id, age, salary, credit_score, loan_amount,
        loan_status, probability, created_at.
    """
    limit = min(limit, 100)
    db = SessionLocal()
    try:
        rows = db.execute(
            text(
                """
                SELECT id, age, salary, credit_score, loan_amount,
                       loan_status, probability, created_at
                FROM predictions
                ORDER BY id DESC
                LIMIT :n
                """
            ),
            {"n": limit},
        ).fetchall()
        return [dict(r._mapping) for r in rows]
    finally:
        db.close()


# ── Tool 4: Feature importance ────────────────────────────────────────────────

@mcp.tool()
def feature_importance() -> dict:
    """
    Return the logistic regression model's per-feature coefficients (log-odds).
    A larger positive coefficient means the feature strongly pushes toward approval;
    a large negative coefficient means it pushes toward rejection.

    Returns:
        dict mapping feature name → coefficient (float).
        Features: age, salary, credit_score, loan_amount.
    """
    return PredictionService.feature_importance()


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
