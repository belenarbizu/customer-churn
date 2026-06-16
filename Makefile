.PHONY: install train-all train-baseline train-rf train-lgbm serve stop test drift reference

# ── Entorno ────────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

# ── Entrenamiento ──────────────────────────────────────────────────────────────
train-baseline:
	python src/train.py -b

train-rf:
	python src/train.py -m

train-lgbm:
	python src/train.py -l

train-all:
	python src/train.py -b
	python src/train.py -m
	python src/train.py -l

reference:
	python src/export_reference.py

# ── API ────────────────────────────────────────────────────────────────────────
serve:
	docker-compose up --build

stop:
	docker-compose down

# ── Tests ──────────────────────────────────────────────────────────────────────
test:
	pytest tests/test_api.py -v

# ── Monitorización ─────────────────────────────────────────────────────────────
drift:
	python src/run_drift.py

# ── MLflow ─────────────────────────────────────────────────────────────────────
mlflow:
	mlflow ui --port 5000
