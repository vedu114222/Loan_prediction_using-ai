"""
FastAPI router for loan prediction.
POST /api/predict-loan — run ML inference and persist result to predictions table.
GET  /api/predictions  — fetch recent predictions with optional limit.
GET  /api/feature-importance — return model coefficients.
"""

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text

from app.models.request_models import LoanRequest
from app.services.prediction_service import PredictionService
from app.utils.db import SessionLocal

router = APIRouter(tags=["Predictions"])


@router.post("/predict-loan", summary="Run ML inference on a loan application")
def predict_loan(request: LoanRequest):
    """
    Submit applicant data and receive an instant approval decision.
    The result is also persisted to the `predictions` table.
    """
    result = PredictionService.predict(request)

    db = SessionLocal()
    try:
        db.execute(
            text(
                """
                INSERT INTO predictions
                    (age, salary, credit_score, loan_amount, loan_status, probability)
                VALUES
                    (:age, :salary, :credit_score, :loan_amount, :loan_status, :probability)
                """
            ),
            {**request.model_dump(), "loan_status": result["loan_status"], "probability": result["probability"]},
        )
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB write failed: {exc}") from exc
    finally:
        db.close()

    return result


@router.get("/predictions", summary="Fetch recent predictions from the database")
def get_predictions(limit: int = Query(10, ge=1, le=100)):
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


@router.get("/feature-importance", summary="Return model feature coefficients")
def feature_importance():
    """Return log-odds coefficients for each input feature."""
    return PredictionService.feature_importance()
