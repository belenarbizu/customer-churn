# src/export_reference.py
import kagglehub, os, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

data_dir = Path("../data")
data_dir.mkdir(exist_ok=True)

download_path = kagglehub.dataset_download("blastchar/telco-customer-churn", output_dir=str(data_dir))
csv = next(f for f in os.listdir(download_path) if f.endswith(".csv"))
df = pd.read_csv(os.path.join(download_path, csv))

X = df.drop(columns=["Churn", "customerID"])
X_train, _ = train_test_split(X, test_size=0.2, random_state=42)
X_train.to_csv(data_dir / "reference.csv", index=False)
print(f"✓ Guardado: {data_dir / 'reference.csv'} ({len(X_train)} filas)")
