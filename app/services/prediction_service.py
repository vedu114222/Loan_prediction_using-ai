"""
Prediction service — loads the trained model once at import time and exposes
two prediction helpers: one for Pydantic request objects, one for raw values
(used by the MCP server).
"""

import joblib
import numpy as np
from pathlib import Path

_MODEL_PATH = Path(__file__).resolve().parent.parent / "ml" / "loan_model.pkl"

try:
    _model = joblib.load(_MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(
        f"Model not found at {_MODEL_PATH}. "
        "Run `python app/ml/train_model.py` first."
    )


class PredictionService:
    @staticmethod
    def predict(data) -> dict:
        """Accept a LoanRequest pydantic object and return a prediction dict."""
        return PredictionService.predict_raw(
            data.age, data.salary, data.credit_score, data.loan_amount
        )

    @staticmethod
    def predict_raw(
        age: int | float,
        salary: float,
        credit_score: int | float,
        loan_amount: float,
    ) -> dict:
        """Accept raw numeric values and return a prediction dict."""
        features = np.array([[age, salary, credit_score, loan_amount]], dtype=float)
        prediction = int(_model.predict(features)[0])
        probability = float(_model.predict_proba(features)[0][1])
        return {
            "loan_status": "Approved" if prediction == 1 else "Rejected",
            "probability": round(probability, 4),
            "approved": bool(prediction),
        }

    @staticmethod
    def feature_importance() -> dict:
        """Return per-feature coefficients (log-odds) from the logistic regression."""
        feature_names = ["age", "salary", "credit_score", "loan_amount"]
        coefs = _model.coef_[0].tolist()
        return dict(zip(feature_names, [round(c, 6) for c in coefs]))
