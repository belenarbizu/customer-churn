import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


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
    model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Mean AUC:", scores.mean())


def baseline_model(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, y_proba))


def main():
    file_path = 'data\\processed_data.csv'
    df = open_file(file_path)
    X_train, X_test, y_train, y_test = split_data(df)
    baseline_model(X_train, y_train, X_test, y_test)
    create_model(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()