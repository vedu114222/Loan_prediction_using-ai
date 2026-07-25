"""
LangChain tool definitions — thin wrappers around the existing service functions.

These are the SAME functions exposed via the MCP server, just decorated with
@tool so LangGraph can discover and call them automatically during ReAct reasoning.
"""

from langchain_core.tools import tool

from app.services.prediction_service import PredictionService
from app.rag.retriever import search_policy_chunks
from app.utils.db import SessionLocal
from sqlalchemy import text


@tool
def predict_loan(age: int, salary: float, credit_score: int, loan_amount: float) -> dict:
    """
    Run the ML model on a loan applicant's data and return an approval decision.

    Use this tool whenever you need to evaluate whether a specific applicant
    would be approved or rejected, and with what probability.

    Args:
        age:          Applicant age in years (18-100).
        salary:       Annual gross salary in USD.
        credit_score: FICO credit score (300-850).
        loan_amount:  Requested loan amount in USD.

    Returns:
        Dict with loan_status ('Approved' or 'Rejected'), probability (0.0-1.0),
        and approved (bool).
    """
    return PredictionService.predict_raw(age, salary, credit_score, loan_amount)


@tool
def search_policy(query: str, top_k: int = 4) -> list[dict]:
    """
    Semantic search over the bank's underwriting policy documents.

    ALWAYS call this tool before explaining a loan decision. It retrieves the
    actual policy passages relevant to your query so your answer is grounded
    in real policy language, not general knowledge.

    Args:
        query: Natural-language question, e.g. 'minimum credit score for approval'
               or 'DTI limit for conventional mortgage' or 'compensating factors'.
        top_k: Number of policy passages to return (1-10, default 4).

    Returns:
        List of dicts with: content (policy text), source (filename), score (0-1).
        Higher score = more semantically relevant to your query.
    """
    return search_policy_chunks(query, min(top_k, 10))


@tool
def get_history(limit: int = 10) -> list[dict]:
    """
    Fetch the most recent loan predictions stored in PostgreSQL.

    Use this to spot patterns across recent applications, compare a current
    applicant against recent cases, or summarise approval trends.

    Args:
        limit: Number of records to return (1-100, default 10).

    Returns:
        List of dicts with: id, age, salary, credit_score, loan_amount,
        loan_status, probability, created_at.
    """
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
            {"n": min(limit, 100)},
        ).fetchall()
        return [dict(r._mapping) for r in rows]
    finally:
        db.close()


@tool
def feature_importance() -> dict:
    """
    Return the logistic regression model's per-feature coefficients (log-odds).

    Use this to explain WHICH applicant factors drive approval or rejection.
    A larger positive value = stronger push toward approval.
    A larger negative value = stronger push toward rejection.

    Returns:
        Dict mapping feature name to coefficient: {age, salary, credit_score, loan_amount}.
    """
    return PredictionService.feature_importance()


# Convenience list — passed directly to create_react_agent
ALL_TOOLS = [predict_loan, search_policy, get_history, feature_importance]
