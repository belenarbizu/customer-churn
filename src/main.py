import os
import sys
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Literal
from pathlib import Path
from datetime import date
from fastapi.responses import FileResponse, JSONResponse

# Añadir el directorio actual al path para que joblib encuentre el Preprocessor
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from data_processing import Preprocessor

app = FastAPI(
    title="Telco Customer Churn API",
    description="Predice la probabilidad de abandono de clientes de telecomunicaciones.",
    version="1.0.0"
)

# --- Rutas de modelos
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
LOG_PATH = BASE_DIR / "logs" / "predictions.csv"
LOG_PATH.parent.mkdir(exist_ok=True)

def load_model(name: str):
    path = MODELS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {path}")
    return joblib.load(path)

# Carga los modelos al arrancar (los que existan)
models = {}
for model_file, model_key in [
    ("baseline_model.pkl", "baseline"),
    ("model.pkl", "random_forest"),
    ("lightgbm_model.pkl", "lightgbm"),
]:
    try:
        models[model_key] = load_model(model_file)
        print(f"✓ Modelo cargado: {model_key}")
    except FileNotFoundError:
        print(f"✗ Modelo no encontrado, se omite: {model_key}")


# --- Esquema de entrada
# Basado en el dataset Telco Customer Churn de Kaggle
class CustomerData(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35,
                "TotalCharges": 844.20
            }
        }
    }


# --- Endpoints

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "models_loaded": list(models.keys())}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "models_available": list(models.keys())}


@app.post("/predict/{model_name}", tags=["Prediction"])
def predict(model_name: str, data: CustomerData):
    """
    Predice la probabilidad de churn para un cliente.

    - **model_name**: `baseline`, `random_forest` o `lightgbm`
    """
    if model_name not in models:
        available = list(models.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Modelo '{model_name}' no disponible. Disponibles: {available}"
        )

    model = models[model_name]
    # Orden de columnas igual que en entrenamiento
    FEATURE_ORDER = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges"
    ]

    row = data.model_dump()

    # Fuerza tipos correctos: numéricas como float, resto como str
    numeric_cols = {"SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"}
    for key in row:
        if key in numeric_cols:
            row[key] = float(row[key])
        else:
            row[key] = str(row[key])

    df = pd.DataFrame([row])[FEATURE_ORDER]

    try:
        proba = float(model.predict_proba(df)[0][1])
        prediction = "Yes" if proba >= 0.5 else "No"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

    _log_prediction(data, proba, prediction, model_name)

    return {
        "churn": prediction,
        "churn_probability": round(proba, 4),
        "model": model_name
    }


@app.post("/predict", tags=["Prediction"])
def predict_default(data: CustomerData):
    """Predice usando el mejor modelo disponible (lightgbm > random_forest > baseline)."""
    for preferred in ["lightgbm", "random_forest", "baseline"]:
        if preferred in models:
            return predict(preferred, data)
    raise HTTPException(status_code=503, detail="No hay modelos cargados.")


@app.get("/models", tags=["Info"])
def list_models():
    """Lista los modelos disponibles."""
    return {"models": list(models.keys())}


@app.get("/drift-report", tags=["Monitoring"])
def get_drift_report():
    """Devuelve el último reporte HTML de drift generado."""
    report_path = Path(__file__).parent.parent / "reports" / "drift_report.html"
    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No hay reporte generado todavía. Ejecuta /drift-run primero."
        )
    return FileResponse(str(report_path), media_type="text/html")


@app.post("/drift-run", tags=["Monitoring"])
def trigger_drift(days: int = 7):
    """Ejecuta el análisis de drift manualmente."""
    try:
        from run_drift import run_drift
        result = run_drift(days=days)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Logging para Evidently (drift monitoring)

def _log_prediction(data: CustomerData, proba: float, prediction: str, model_name: str):
    row = data.model_dump()
    row["prediction"] = prediction
    row["churn_probability"] = proba
    row["model"] = model_name
    row["date"] = str(date.today())

    df_row = pd.DataFrame([row])
    df_row.to_csv(
        LOG_PATH,
        mode="a",
        header=not LOG_PATH.exists(),
        index=False,
        quoting=1
    )
