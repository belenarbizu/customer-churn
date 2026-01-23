import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
import joblib
from data_processing import Preprocessor
from sklearn.pipeline import Pipeline
import json


param_grid = {
    'classifier__n_estimators': [200, 300, 400, 500],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10]
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
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, y_proba))


def save_model(model, file_path):
    """Saves the model to a file."""
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    joblib.dump(model, file_path)


def save_info(params, report, score):
    """Saves the model to a file."""
    if not os.path.exists(os.path.dirname('models\\best_params.json')):
        os.makedirs(os.path.dirname('models\\best_params.json'))
    if not os.path.exists(os.path.dirname('models\\metrics.json')):
        os.makedirs(os.path.dirname('models\\metrics.json'))

    with open('models\\best_params.json', 'w') as f:
        json.dump(params, f, indent=4)

    metrics = {
        "accuracy": report["accuracy"],
        "roc_auc": score,
        "f1_macro": report["macro avg"]["f1-score"],
        "precision_yes": report["Yes"]["precision"],
        "recall_yes": report["Yes"]["recall"],
        "f1_yes": report["Yes"]["f1-score"]
    }

    with open('models\\metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)


def save_predictions(y_test, y_pred, y_proba):
    y_test = y_test.map({'No': 0, 'Yes': 1}).reset_index(drop=True)
    y_pred = pd.Series(y_pred).map({'No': 0, 'Yes': 1}).reset_index(drop=True)

    results = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba
    })

    if not os.path.exists('models'):
        os.makedirs('models')
    results.to_csv('models\\predictions.csv', index=False)


def main():
    file_path = 'data\\Telco-Customer-Churn.csv'
    df = open_file(file_path)
    X_train, X_test, y_train, y_test = split_data(df)
    # baseline_model(X_train, y_train, X_test, y_test)
    best_model, best_params, report, best_score, y_pred, y_proba = create_model(X_train, y_train, X_test, y_test)
    save_predictions(y_test, y_pred, y_proba)
    save_model(best_model, 'models\\model.pkl')
    save_info(best_params, report, best_score)


if __name__ == "__main__":
    main()