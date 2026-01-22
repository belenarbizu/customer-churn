import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
import joblib


param_grid = {
    'n_estimators': [200, 300, 400, 500],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
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
    cv = RandomizedSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'), 
                            param_grid, n_iter=10, cv=5, scoring='roc_auc', random_state=42, n_jobs=-1)
    cv.fit(X_train, y_train)
    return cv.best_estimator_


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


def main():
    file_path = 'data\\processed_data.csv'
    df = open_file(file_path)
    X_train, X_test, y_train, y_test = split_data(df)
    # baseline_model(X_train, y_train, X_test, y_test)
    best_model = create_model(X_train, y_train, X_test, y_test)
    save_model(best_model, 'models\\model.pkl')


if __name__ == "__main__":
    main()