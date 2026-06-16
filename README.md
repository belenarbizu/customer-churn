# Customer Churn Prediction — MLOps Pipeline

End-to-end MLOps project for predicting customer churn: model training with experiment tracking, REST API in production, containerization, and data drift monitoring.

**Dataset**: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) · **Stack**: Python · FastAPI · Docker · MLflow · LightGBM · scikit-learn · Evidently AI

🚀 **Live API**: [customer-churn-api-lhqb.onrender.com/docs](https://customer-churn-api-lhqb.onrender.com/docs)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Training                            │
│   train.py  ──►  MLflow tracking  ──►  models/*.pkl         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Production (Docker + Render)             │
│                                                             │
│   FastAPI /predict  ──►  sklearn Pipeline + LightGBM        │
│         │                                                   │
│         └──►  logs/predictions.csv                          │
│                          │                                  │
│                          ▼                                  │
│              Evidently AI drift monitor                     │
│              /drift-run  ──►  /drift-report                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    CI (GitHub Actions)                      │
│         Daily drift check · pytest on every push            │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Overview

This project focuses on predicting customer churn using classical machine learning models and ensemble methods. The main goal is to correctly identify customers who are likely to churn — a minority class problem — making recall and F1-score for the positive class especially important.

The project includes:

- Data preprocessing with a unified sklearn pipeline
- A baseline model for reference
- Ensemble models (Random Forest and LightGBM)
- Model evaluation, visualization, and experiment tracking with MLflow
- REST API with FastAPI, containerized with Docker
- Automated data drift monitoring with Evidently AI
- Test suite with pytest
- Deployment on Render (free tier)
- Daily CI via GitHub Actions

---

## Data Preprocessing

The preprocessing step was designed as part of a pipeline to ensure reproducibility and avoid data leakage.

Key decisions:
- Numerical features were not normally distributed, but this did not affect model performance as tree-based models were used.
- Binary categorical features (`Yes`/`No`) were mapped to `1`/`0`.
- Multi-class categorical features were encoded using One-Hot Encoding.
- No significant outliers were detected in the dataset.

All steps are encapsulated in a custom `Preprocessor` class (`data_processing.py`) that integrates cleanly into the sklearn pipeline.

---

## Models

**Baseline — Logistic Regression**
Used to establish a minimum performance reference.

**Random Forest (Bagging)**
Provided strong churn detection performance, especially in terms of recall and ROC-AUC. Proved to be robust and stable for this dataset.

**LightGBM (Boosting)**
Used to compare bagging vs. boosting approaches. Improved ROC-AUC and recall when properly tuned with class imbalance handling.

### Results

| Model | ROC-AUC | F1 (churn) | Recall (churn) |
|---|---|---|---|
| Logistic Regression (baseline) | 0.84 | 0.61 | 0.56 |
| Random Forest | 0.84 | 0.63 | 0.77 |
| LightGBM | 0.84 | 0.63 | 0.79 |

All three models are available via the API. LightGBM is used by default.

---

## Evaluation Metrics

Since churn is the minority class, model selection focused on:

- **Recall (Yes)** — minimizing false negatives (missed churners)
- **F1-score (Yes)** — balance between precision and recall for the churn class
- **ROC-AUC** — overall class separability

Accuracy alone was not considered sufficient for model selection given the class imbalance.

---

## Project Structure

```
customer-churn/
├── src/
│   ├── train.py              # Training and MLflow tracking
│   ├── data_processing.py    # Preprocessor (OneHotEncoder + cleaning)
│   ├── visualization.py      # Plots: ROC curve, confusion matrix, feature importance
│   ├── main.py               # FastAPI app
│   ├── run_drift.py          # Drift analysis with Evidently
│   └── export_reference.py   # Exports reference dataset (run once)
├── tests/
│   └── test_api.py           # pytest test suite (24 tests)
├── models/                   # Trained .pkl models
├── data/
│   └── reference.csv         # Reference dataset for drift monitoring
├── logs/
│   └── predictions.csv       # Production prediction log (auto-generated)
├── reports/
│   └── drift_report.html     # Latest drift report (auto-generated)
├── .github/
│   └── workflows/
│       └── drift_check.yml   # Daily drift check CI
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
├── setup.py
└── setup.sh
```

---

## Getting Started

### Requirements

- Python 3.11+
- Docker Desktop
- Kaggle account (dataset is downloaded automatically via `kagglehub`)
- `make` (optional but recommended)

### 1. Clone the repository

```bash
git clone https://github.com/belenarbizu/customer-churn.git
cd customer-churn
```

### 2. Set up the environment

```bash
make install
```

Or manually:

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Train the models

```bash
make train-all         # trains all three models
# or individually:
make train-baseline    # Logistic Regression
make train-rf          # Random Forest
make train-lgbm        # LightGBM
```

Artifacts are saved to `models/` and experiments tracked in MLflow.

### 4. Generate visualizations

```bash
python src/visualization.py
```

Plots are saved to `images/`: confusion matrix, ROC curve, and feature importances.

### 5. View experiments in MLflow

```bash
make mlflow
```

Open `http://localhost:5000` to compare metrics across runs.

### 6. Export the reference dataset (once)

Required for drift monitoring. Only needs to be run once:

```bash
make reference
```

Generates `data/reference.csv` from the training split.

### 7. Start the API with Docker

```bash
make serve
```

This starts two services:
- **API**: `http://localhost:8000`
- **MLflow**: `http://localhost:5000`

To stop:
```bash
make stop
```

### 8. Run the tests

```bash
make test
```

Runs 24 pytest tests covering endpoints, data validation, the preprocessor, and integration flows.

---

## API Usage

Interactive documentation available at:
- **Local**: `http://localhost:8000/docs`
- **Production**: `https://customer-churn-api.onrender.com/docs`

> Note: the free Render instance sleeps after 15 minutes of inactivity. The first request may take ~30 seconds to wake up.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/health` | Service status and loaded models |
| GET | `/models` | List available models |
| POST | `/predict` | Predict with best model (LightGBM) |
| POST | `/predict/{model_name}` | Predict with a specific model |
| POST | `/drift-run` | Run drift analysis |
| GET | `/drift-report` | View HTML drift report in browser |

Available model names: `baseline`, `random_forest`, `lightgbm`.

### Prediction example

```bash
curl -X POST "https://customer-churn-api.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

Response:

```json
{
  "churn": "Yes",
  "churn_probability": 0.7832,
  "model": "lightgbm"
}
```

---

## Drift Monitoring

Every prediction is automatically logged to `logs/predictions.csv`. Evidently AI compares this production data against the reference dataset to detect distribution shifts.

### Run analysis

```bash
# Via API
curl -X POST "http://localhost:8000/drift-run"

# Or directly with Python
make drift
```

### View the report

Open in your browser:

```
http://localhost:8000/drift-report
```

The report includes:
- Feature distribution comparison (reference vs production)
- Automatic pass/fail tests per column
- Alert if more than 2 columns show drift

### Automated daily analysis

The workflow `.github/workflows/drift_check.yml` runs the analysis every day at 08:00 UTC and uploads the report as a GitHub Actions artifact.

---

## Available Make Commands

```bash
make install        # install dependencies
make train-all      # train all three models
make train-baseline # train Logistic Regression
make train-rf       # train Random Forest
make train-lgbm     # train LightGBM
make reference      # export reference.csv for drift monitoring
make serve          # start API + MLflow with Docker
make stop           # stop Docker services
make test           # run pytest suite
make drift          # run drift analysis
make mlflow         # open MLflow UI
```

---

## Conclusion

Logistic Regression serves as a useful baseline but is insufficient for churn detection on its own.

Random Forest offers the best stability for this dataset, especially in recall and ROC-AUC.

LightGBM demonstrates the strengths of boosting and provides competitive results when properly tuned.

Metric selection is critical when dealing with imbalanced data — accuracy alone is misleading.

---

## Tech Stack

| Category | Tools |
|---|---|
| Modeling | scikit-learn · LightGBM · pandas · numpy |
| Experiment tracking | MLflow |
| API | FastAPI · uvicorn · pydantic |
| Containerization | Docker · Docker Compose |
| Drift monitoring | Evidently AI |
| Testing | pytest |
| CI/CD | GitHub Actions |
| Deployment | Render |
