import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import joblib
from data_processing import Preprocessor
from sklearn.pipeline import Pipeline
import json
import mlflow
import mlflow.sklearn
from lightgbm import LGBMClassifier
import argparse
import kagglehub


param_grid = {
    'classifier__n_estimators': [200, 300, 400, 500],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10]
}


lightgbm_param_grid = {
    'classifier__num_leaves': [31, 50, 100],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__n_estimators': [200, 300, 400]
}


def open_file(file_path):
    """Opens a file and returns its contents."""
    df = pd.read_csv(file_path)
    return df


def split_data(df, test_size=0.2, random_state=42):
    """Splits the data into training and testing sets."""
    X = df.drop(['Churn', 'customerID'], axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    return X_train, X_test, y_train, y_test


def lightgbm_model(X_train, y_train, X_test, y_test):
    pipeline = Pipeline(steps=[
        ('preprocessor', Preprocessor()),
        ('classifier', LGBMClassifier(random_state=42, class_weight='balanced'))
    ])
    cv = RandomizedSearchCV(pipeline, lightgbm_param_grid, n_iter=10, cv=5, scoring='roc_auc', random_state=42, n_jobs=-1)
    cv.fit(X_train, y_train)
    y_pred = cv.predict(X_test)
    y_proba = cv.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    score = roc_auc_score(y_test, y_proba)
    return cv.best_estimator_, {'model': 'LGBMClassifier'}, report, score, y_pred, y_proba


def create_model(X_train, y_train, X_test, y_test):
    pipeline = Pipeline(steps=[
        ('preprocessor', Preprocessor()),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])
    cv = RandomizedSearchCV(pipeline, param_grid, n_iter=10, cv=5, scoring='roc_auc', random_state=42, n_jobs=-1)
    cv.fit(X_train, y_train)
    y_pred = cv.predict(X_test)
    y_proba = cv.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    return cv.best_estimator_, cv.best_params_, report, cv.best_score_, y_pred, y_proba


def baseline_model(X_train, y_train, X_test, y_test):
    pipeline = Pipeline(steps=[
        ('preprocessor', Preprocessor()),
        ('classifier', LogisticRegression(max_iter=5000))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    score = roc_auc_score(y_test, y_proba)
    return pipeline, {'model': 'LogisticRegression', 'max_iter': 5000}, report, score, y_pred, y_proba


def save_model(model, file_path):
    """Saves the model to a file."""
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    joblib.dump(model, file_path)


def save_info(params, report, score, metrics_path, params_path):
    """Saves the model to a file."""
    if not os.path.exists(os.path.dirname(params_path)):
        os.makedirs(os.path.dirname(params_path))
    if not os.path.exists(os.path.dirname(metrics_path)):
        os.makedirs(os.path.dirname(metrics_path))

    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)

    metrics = {
        "accuracy": report["accuracy"],
        "roc_auc": score,
        "f1_macro": report["macro avg"]["f1-score"],
        "precision_yes": report["Yes"]["precision"],
        "recall_yes": report["Yes"]["recall"],
        "f1_yes": report["Yes"]["f1-score"]
    }

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)


def save_predictions(y_test, y_pred, y_proba, file_path):
    y_test = y_test.map({'No': 0, 'Yes': 1}).reset_index(drop=True)
    y_pred = pd.Series(y_pred).map({'No': 0, 'Yes': 1}).reset_index(drop=True)

    # Asegurar que el directorio de destino exista
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    results = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba
    })
    results.to_csv(file_path, index=False)


def mlflow_model_logging(model, report, score, params):
    mlflow.set_experiment("Telco Churn Prediction")
    with mlflow.start_run(run_name="RandomForestClassifier"):
        mlflow.log_params({
            'model': 'RandomForest',
            'n_estimators': params['classifier__n_estimators'],
            'max_depth': params['classifier__max_depth'],
            'min_samples_split': params['classifier__min_samples_split']
        })

        mlflow.log_metrics({
            'accuracy': report['accuracy'],
            'roc_auc': score,
            'f1_macro': report['macro avg']['f1-score'],
            'precision_yes': report['Yes']['precision'],
            'recall_yes': report['Yes']['recall'],
            'f1_yes': report['Yes']['f1-score']
        })

        mlflow.sklearn.log_model(model, "model")
        try:
            mlflow.log_artifact("..\\models\\metrics.json")
            mlflow.log_artifact("..\\images\\roc_curve.png")
            mlflow.log_artifact("..\\images\\confusion_matrix.png")
            mlflow.log_artifact("..\\images\\feature_importances.png")
            mlflow.log_artifact("..\\images\\decision_tree.png")
        except Exception as e:
            print(f"Error logging artifact: {e}")


def mlflow_baseline(model, report, score, params, model_name):
    mlflow.set_experiment("Telco Churn Prediction")
    name = model_name + "_model"
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)

        mlflow.log_metrics({
            'accuracy': report['accuracy'],
            'roc_auc': score,
            'f1_macro': report['macro avg']['f1-score'],
            'precision_yes': report['Yes']['precision'],
            'recall_yes': report['Yes']['recall'],
            'f1_yes': report['Yes']['f1-score']
        })

        mlflow.sklearn.log_model(model, model_name)
        try:
            mlflow.log_artifact(f"..\\models\\metrics_{model_name}.json")
        except Exception as e:
            print(f"Error logging artifact: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--baseline', action='store_true', help='Run baseline model only')
    parser.add_argument('-m', '--model', action='store_true', help='Run advanced model only')
    parser.add_argument('-l', '--lightgbm', action='store_true', help='Run LightGBM model only')
    args = parser.parse_args()

    data_dir = os.path.join('..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    download_path = kagglehub.dataset_download("blastchar/telco-customer-churn", output_dir=data_dir)

    csv_files = [f for f in os.listdir(download_path) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No se encontró ningún archivo CSV en {download_path}")
    
    file_to_open = os.path.join(download_path, csv_files[0])
    df = open_file(file_to_open)
    X_train, X_test, y_train, y_test = split_data(df)

    if args.baseline:
        base_model, base_params, base_report, base_score, y_pred, y_proba = baseline_model(X_train, y_train, X_test, y_test)
        save_predictions(y_test, y_pred, y_proba, '..\\models\\predictions_baseline.csv')
        save_model(base_model, '..\\models\\baseline_model.pkl')
        save_info(base_params, base_report, base_score, '..\\models\\metrics_baseline.json', '..\\models\\params_baseline.json')
        mlflow_baseline(base_model, base_report, base_score, base_params, 'baseline')
    if args.model:
        best_model, best_params, report, best_score, y_pred, y_proba = create_model(X_train, y_train, X_test, y_test)
        save_predictions(y_test, y_pred, y_proba, '..\\models\\predictions.csv')
        save_model(best_model, '..\\models\\model.pkl')
        save_info(best_params, report, best_score, '..\\models\\metrics.json', '..\\models\\params.json')
        mlflow_model_logging(best_model, report, best_score, best_params)
    if args.lightgbm:
        lgbm_model, lgbm_params, lgbm_report, lgbm_score, y_pred, y_proba = lightgbm_model(X_train, y_train, X_test, y_test)
        save_predictions(y_test, y_pred, y_proba, '..\\models\\predictions_lightgbm.csv')
        save_model(lgbm_model, '..\\models\\lightgbm_model.pkl')
        save_info(lgbm_params, lgbm_report, lgbm_score, '..\\models\\metrics_lightgbm.json', '..\\models\\params_lightgbm.json')
        mlflow_baseline(lgbm_model, lgbm_report, lgbm_score, lgbm_params, 'lightgbm')


if __name__ == "__main__":
    main()