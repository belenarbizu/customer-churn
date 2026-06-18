import pandas as pd
import sys
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).parent))

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfDriftedColumns

REFERENCE_PATH = Path(__file__).parent.parent / "data" / "reference.csv"
LOG_PATH       = Path(__file__).parent.parent / "logs" / "predictions.csv"
REPORT_PATH    = Path(__file__).parent.parent / "reports" / "drift_report.html"

# Columnas de features (sin prediction, date, model)
FEATURE_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

def load_reference():
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset de referencia en {REFERENCE_PATH}."
        )
    df = pd.read_csv(REFERENCE_PATH, quoting=0, on_bad_lines='skip')[FEATURE_COLS]

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce').fillna(0)
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors='coerce')
    df["tenure"] = pd.to_numeric(df["tenure"], errors='coerce')
    df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors='coerce')

    return df

def load_current(days: int = 7):
    if not LOG_PATH.exists():
        raise FileNotFoundError("No hay logs de predicciones todavía.")
    df = pd.read_csv(LOG_PATH, quoting=0, on_bad_lines='skip')
    df["date"] = pd.to_datetime(df["date"])
    cutoff = pd.Timestamp(date.today()) - timedelta(days=days)
    current = df[df["date"] >= cutoff][FEATURE_COLS].copy()

    if current.empty:
        raise ValueError(f"No hay predicciones en los últimos {days} días.")

    current["TotalCharges"] = pd.to_numeric(current["TotalCharges"], errors='coerce').fillna(0)
    current["MonthlyCharges"] = pd.to_numeric(current["MonthlyCharges"], errors='coerce')
    current["tenure"] = pd.to_numeric(current["tenure"], errors='coerce')
    current["SeniorCitizen"] = pd.to_numeric(current["SeniorCitizen"], errors='coerce')

    return current

def run_drift(days: int = 7) -> dict:
    reference = load_reference()
    current   = load_current(days)

    REPORT_PATH.parent.mkdir(exist_ok=True)

    # Reporte HTML visual
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html(str(REPORT_PATH))

    # Test suite para alertas (falla si >2 columnas con drift)
    suite = TestSuite(tests=[TestNumberOfDriftedColumns(lt=3)])
    suite.run(reference_data=reference, current_data=current)
    results = suite.as_dict()

    n_drifted = sum(
        1 for t in results["tests"]
        if "drifted" in t.get("name", "").lower() or t.get("status") == "FAIL"
    )

    return {
        "all_passed":  results["summary"]["all_passed"],
        "n_drifted":   n_drifted,
        "total_cols":  len(FEATURE_COLS),
        "days_window": days,
        "report_path": str(REPORT_PATH),
    }

if __name__ == "__main__":
    result = run_drift()
    print(result)
    if not result["all_passed"]:
        print("⚠️  ALERTA: drift detectado")
        sys.exit(1)
