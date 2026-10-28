
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "artifacts" / "final_bug_fix_effort_model.joblib"

model_bundle = joblib.load(MODEL_PATH)
model = model_bundle["model"]
preprocessor = model_bundle["preprocessor"]
feature_columns = model_bundle["feature_columns"]

app = FastAPI(
    title="Bug Fix Effort Predictor API",
    description="Predicts software bug resolution time and estimates risk.",
    version="1.0.0"
)

class BugReport(BaseModel):
    reporter_accessible: int = 1
    cclist_accessible: int = 1
    everconfirmed: int = 1
    restrict_comments: int = 0
    creation_year: int = 2024
    creation_month: int = 6
    creation_day: int = 15
    creation_dayofweek: int = 5
    creation_quarter: int = 2
    classification: str = "Other"
    product: str = "Other"
    component: str = "Other"
    version: str = "Other"
    rep_platform: str = "Other"
    op_sys: str = "Other"
    priority: str = "P3"
    bug_severity: str = "normal"
    reporter: str = "unknown"
    comment_sort_order: str = "oldest_to_newest"
    project: str = "Mozilla"
    mentor: str = "unknown"

@app.get("/")
def home():
    return {
        "message": "Bug Fix Effort Predictor API is running",
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": model_bundle.get("model_name", "ML model")}

@app.post("/predict")
def predict_bug(report: BugReport):
    try:
        data = report.model_dump() if hasattr(report, "model_dump") else report.dict()
        input_df = pd.DataFrame([data])[feature_columns]
        transformed = preprocessor.transform(input_df)

        predicted_log_days = float(model.predict(transformed)[0])
        predicted_days = max(0.0, float(np.expm1(predicted_log_days)))

        if predicted_days <= 7:
            risk = "Low"
            advice = "Review the fix, run relevant tests, and verify the resolution."
        elif predicted_days <= 30:
            risk = "Moderate"
            advice = "Break the fix into smaller tasks and monitor progress."
        elif predicted_days <= 90:
            risk = "High"
            advice = "Investigate dependencies, assign clear ownership, and review blockers."
        else:
            risk = "Very High"
            advice = "Review the root cause, dependencies, scope, and resource requirements."

        return {
            "predicted_resolution_days": round(predicted_days, 2),
            "risk_category": risk,
            "recommended_action": advice,
            "note": "Risk categories and advice are rule-based, not Gemini-generated."
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
