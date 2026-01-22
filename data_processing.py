import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin


categorical_cols = [
    'gender',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaymentMethod'
]


class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    def fit(self, X, y=None):
        X_clean = self._clean_data(X)
        # Only encode columns that exist in the dataframe
        self.encoded_cols_ = [col for col in categorical_cols if col in X_clean.columns]
        if self.encoded_cols_:
            self.encoder.fit(X_clean[self.encoded_cols_])
        return self

    def transform(self, X):
        X_clean = self._clean_data(X)
        if not hasattr(self, 'encoded_cols_') or not self.encoded_cols_:
            return X_clean

        encoded_cols = self.encoder.get_feature_names_out(self.encoded_cols_)
        encoded_data = pd.DataFrame(
            self.encoder.transform(X_clean[self.encoded_cols_]),
            columns=encoded_cols,
            index=X_clean.index
        )
        return pd.concat([X_clean.drop(columns=self.encoded_cols_), encoded_data], axis=1)

    def _clean_data(self, df):
        df = df.copy()
        # Convert 'TotalCharges' to numeric, coercing errors to NaN, then fill with 0
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce').fillna(0)
        
        # Map Yes/No to 1/0
        for col in df.columns:
            if df[col].dtype == 'object' and set(df[col].dropna().unique()).issubset({'Yes', 'No'}):
                df[col] = df[col].map({'Yes': 1, 'No': 0})

        # Map PaymentMethod
        if "PaymentMethod" in df.columns:
            df["PaymentMethod"] = df["PaymentMethod"].map({
                'Electronic check': 'Electronic Check',
                'Mailed check': 'Mailed Check',
                'Bank transfer (automatic)': 'Bank Transfer',
                'Credit card (automatic)': 'Credit Card'
            })
        return df
