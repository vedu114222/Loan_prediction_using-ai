"""
Train a logistic regression model on synthetic loan data and save it to app/ml/loan_model.pkl.
Run this once before starting the API:
    python app/ml/train_model.py
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── Reproducible synthetic dataset ────────────────────────────────────────────
rng = np.random.default_rng(42)
n = 5_000

age = rng.integers(21, 70, n).astype(float)
salary = rng.integers(20_000, 200_000, n).astype(float)
credit_score = rng.integers(300, 850, n).astype(float)
loan_amount = rng.integers(10_000, 750_000, n).astype(float)

# Composite score that loosely mimics underwriting logic
score = (
    0.0008 * salary
    + 0.025 * credit_score
    - 0.00004 * loan_amount
    + 0.015 * age
    - 5.0  # baseline offset
)

# Approval: top 60% by score → approved
threshold = np.percentile(score, 40)
approved = (score > threshold).astype(int)

X = np.column_stack([age, salary, credit_score, loan_amount])

# ── Train / evaluate ───────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, approved, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

print("-- Classification Report --")
print(classification_report(y_test, model.predict(X_test)))

# ── Persist ────────────────────────────────────────────────────────────────────
out_path = Path(__file__).resolve().parent / "loan_model.pkl"
joblib.dump(model, out_path)
print(f"[OK] Saved model -> {out_path}")
