"""
Credit Risk Model — Expected Loss Estimator
---------------------------------------------
Trains a logistic regression model on loan data to predict
probability of default (PD) and expected loss (EL).

Usage:
    python task3/task3.py
"""

import sys
import warnings
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_PATH     = Path(__file__).parent.parent / "data" / "Task 3 and 4_Loan_Data.csv"
RECOVERY_RATE = 0.10
FEATURES      = [
    'credit_lines_outstanding',
    'loan_amt_outstanding',
    'total_debt_outstanding',
    'income',
    'years_employed',
    'fico_score',
]

# --- Module-level model objects (populated by train_model) ---
_scaler = None
_model  = None


def train_model(data_path: Path = DATA_PATH):
    """Load data, train logistic regression, return scaler and model."""
    df = pd.read_csv(data_path)

    X = df[FEATURES]
    y = df['default']

    # Stratified 60/20/20 split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)

    return scaler, model


def _ensure_model():
    """Lazily train the model on first use."""
    global _scaler, _model
    if _scaler is None or _model is None:
        _scaler, _model = train_model()


def expected_loss(
    loan_amt_outstanding: float,
    credit_lines_outstanding: float,
    total_debt_outstanding: float,
    income: float,
    years_employed: float,
    fico_score: float,
) -> dict:
    """
    Estimate the expected loss for a loan.

    Parameters
    ----------
    loan_amt_outstanding : float
        Current outstanding loan balance.
    credit_lines_outstanding : float
        Number of open credit lines.
    total_debt_outstanding : float
        Total debt across all obligations.
    income : float
        Annual income.
    years_employed : float
        Years in current employment.
    fico_score : float
        Credit score.

    Returns
    -------
    dict with keys:
        probability_of_default  - estimated PD (0 to 1)
        expected_loss           - EL = PD x loan_amt x (1 - recovery_rate)
    """
    _ensure_model()

    input_df = pd.DataFrame([{
        'credit_lines_outstanding': credit_lines_outstanding,
        'loan_amt_outstanding':     loan_amt_outstanding,
        'total_debt_outstanding':   total_debt_outstanding,
        'income':                   income,
        'years_employed':           years_employed,
        'fico_score':               fico_score,
    }])

    features_scaled = _scaler.transform(input_df)
    pd_prob = _model.predict_proba(features_scaled)[0][1]
    el      = pd_prob * loan_amt_outstanding * (1 - RECOVERY_RATE)

    return {
        "probability_of_default": round(pd_prob, 4),
        "expected_loss":          round(el, 2),
    }


if __name__ == "__main__":
    print("Training model...")
    _ensure_model()
    print("Done.\n")

    test_cases = [
        {
            "label": "Low risk  (0 credit lines, high income, FICO 750)",
            "loan_amt_outstanding":     3000,
            "credit_lines_outstanding": 0,
            "total_debt_outstanding":   2000,
            "income":                   90000,
            "years_employed":           8,
            "fico_score":               750,
        },
        {
            "label": "Medium risk (2 credit lines, mid income, FICO 600)",
            "loan_amt_outstanding":     5000,
            "credit_lines_outstanding": 2,
            "total_debt_outstanding":   8000,
            "income":                   55000,
            "years_employed":           4,
            "fico_score":               600,
        },
        {
            "label": "High risk  (4 credit lines, low income, FICO 500)",
            "loan_amt_outstanding":     8000,
            "credit_lines_outstanding": 4,
            "total_debt_outstanding":   25000,
            "income":                   30000,
            "years_employed":           1,
            "fico_score":               500,
        },
    ]

    print(f"{'Borrower':<50} {'PD':>8} {'Expected Loss':>15}")
    print("-" * 75)
    for case in test_cases:
        label = case.pop("label")
        result = expected_loss(**case)
        print(f"{label:<50} {result['probability_of_default']:>8.2%} ${result['expected_loss']:>13,.2f}")
