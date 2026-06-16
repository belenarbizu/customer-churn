import sys
import pytest
import pandas as pd
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Payload de ejemplo válido
# ---------------------------------------------------------------------------
VALID_PAYLOAD = {
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


# ---------------------------------------------------------------------------
# Tests de endpoints básicos
# ---------------------------------------------------------------------------

def test_health():
    """El endpoint raíz responde con status 200."""
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "ok"


def test_health_endpoint():
    """El endpoint /health responde correctamente."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_list_models_returns_list():
    """/models devuelve una lista."""
    response = client.get("/models")
    assert response.status_code == 200
    assert "models" in response.json()
    assert isinstance(response.json()["models"], list)


def test_list_models_not_empty():
    """/models tiene al menos un modelo cargado."""
    response = client.get("/models")
    assert len(response.json()["models"]) > 0


# ---------------------------------------------------------------------------
# Tests de predicción con datos válidos
# ---------------------------------------------------------------------------

def test_predict_default_status_200():
    """POST /predict con datos válidos devuelve 200."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200


def test_predict_default_returns_churn_field():
    """La respuesta contiene el campo 'churn'."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert "churn" in response.json()


def test_predict_default_churn_value():
    """El campo 'churn' es 'Yes' o 'No'."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.json()["churn"] in ["Yes", "No"]


def test_predict_default_probability_range():
    """La probabilidad de churn está entre 0 y 1."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    proba = response.json()["churn_probability"]
    assert 0.0 <= proba <= 1.0


def test_predict_default_returns_model_name():
    """La respuesta indica qué modelo se usó."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert "model" in response.json()
    assert response.json()["model"] in ["baseline", "random_forest", "lightgbm"]


# ---------------------------------------------------------------------------
# Tests de predicción por nombre de modelo
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_name", ["baseline", "lightgbm"])
def test_predict_by_model_name(model_name):
    """POST /predict/{model_name} devuelve predicción válida para cada modelo."""
    response = client.post(f"/predict/{model_name}", json=VALID_PAYLOAD)
    # Si el modelo no está cargado se acepta 404, si está cargado debe ser 200
    assert response.status_code in [200, 404]
    if response.status_code == 200:
        data = response.json()
        assert "churn" in data
        assert "churn_probability" in data
        assert 0.0 <= data["churn_probability"] <= 1.0


def test_predict_unknown_model_returns_404():
    """Solicitar un modelo inexistente devuelve 404."""
    response = client.post("/predict/modelo_que_no_existe", json=VALID_PAYLOAD)
    assert response.status_code == 404


def test_predict_unknown_model_error_message():
    """El error 404 incluye un mensaje descriptivo."""
    response = client.post("/predict/modelo_inventado", json=VALID_PAYLOAD)
    assert "detail" in response.json()


# ---------------------------------------------------------------------------
# Tests de validación de datos (422 Unprocessable Entity)
# ---------------------------------------------------------------------------

def test_predict_missing_field_returns_422():
    """Faltar un campo obligatorio devuelve 422."""
    payload_incompleto = VALID_PAYLOAD.copy()
    del payload_incompleto["tenure"]
    response = client.post("/predict", json=payload_incompleto)
    assert response.status_code == 422


def test_predict_invalid_gender_returns_422():
    """Un valor no permitido en 'gender' devuelve 422."""
    payload = VALID_PAYLOAD.copy()
    payload["gender"] = "Unknown"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_negative_tenure_returns_422():
    """Un valor negativo en 'tenure' devuelve 422."""
    payload = VALID_PAYLOAD.copy()
    payload["tenure"] = -1
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_contract_returns_422():
    """Un valor no permitido en 'Contract' devuelve 422."""
    payload = VALID_PAYLOAD.copy()
    payload["Contract"] = "Weekly"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_empty_payload_returns_422():
    """Un payload vacío devuelve 422."""
    response = client.post("/predict", json={})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Tests del preprocesador
# ---------------------------------------------------------------------------

def test_preprocessor_fit_transform():
    """El preprocesador transforma datos válidos sin errores."""
    from data_processing import Preprocessor

    df = pd.DataFrame([VALID_PAYLOAD])
    prep = Preprocessor()
    prep.fit(df)
    result = prep.transform(df)
    assert result is not None
    assert len(result) == 1


def test_preprocessor_total_charges_coerce():
    """TotalCharges vacío o no numérico se convierte a 0."""
    from data_processing import Preprocessor

    df = pd.DataFrame([{**VALID_PAYLOAD, "TotalCharges": " "}])
    prep = Preprocessor()
    prep.fit(df)
    result = prep.transform(df)
    assert result["TotalCharges"].iloc[0] == 0.0


def test_preprocessor_yes_no_mapping():
    """Partner='Yes' se convierte a 1 y 'No' a 0."""
    from data_processing import Preprocessor

    df_yes = pd.DataFrame([{**VALID_PAYLOAD, "Partner": "Yes"}])
    df_no  = pd.DataFrame([{**VALID_PAYLOAD, "Partner": "No"}])

    prep = Preprocessor()
    prep.fit(df_yes)

    result_yes = prep.transform(df_yes)
    result_no  = prep.transform(df_no)

    assert result_yes["Partner"].iloc[0] == 1
    assert result_no["Partner"].iloc[0] == 0


# ---------------------------------------------------------------------------
# Tests de integración
# ---------------------------------------------------------------------------

def test_predict_logs_prediction(tmp_path, monkeypatch):
    """Una predicción exitosa escribe una fila en el log."""
    import main as main_module

    log_file = tmp_path / "predictions.csv"
    monkeypatch.setattr(main_module, "LOG_PATH", log_file)

    client.post("/predict", json=VALID_PAYLOAD)

    assert log_file.exists()
    df = pd.read_csv(log_file)
    assert len(df) == 1
    assert "churn_probability" in df.columns
    assert "date" in df.columns


def test_drift_report_not_found_before_run():
    """Antes de ejecutar /drift-run, /drift-report devuelve 404."""
    # Solo válido si no existe el reporte previo; si existe, se omite
    import main as main_module
    report_path = Path(__file__).parent.parent / "reports" / "drift_report.html"
    if not report_path.exists():
        response = client.get("/drift-report")
        assert response.status_code == 404
