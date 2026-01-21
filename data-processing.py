import pandas as pd
from sklearn.preprocessing import OneHotEncoder


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


def open_file(file_path):
    """Opens a file and returns its contents."""
    df = pd.read_csv(file_path)
    return df


def check_data(df):
    """Checks the data for missing values and prints summary statistics."""
    print("Data Summary:")
    print(df.info())
    print("\nSummary Statistics:")
    print(df.describe())


def change_to_num(df):
    """Changes 'Yes' and 'No' values in a column to 1 and 0. Also converts 'TotalCharges' to numeric."""
    columns_with_yes_no = [col for col in df.columns if df[col].isin(['Yes', 'No']).all()]
    for column_name in columns_with_yes_no:
        df[column_name] = df[column_name].map({'Yes': 1, 'No': 0})

    # Convert 'TotalCharges' to numeric, coercing errors to NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    return df


def data_cleaning(df):
    """Preprocesses the data by checking and converting necessary columns."""
    check_data(df)
    df = change_to_num(df)
    df.drop_duplicates(inplace=True)
    df["PaymentMethod"] = df["PaymentMethod"].map({
        'Electronic check': 'Electronic Check',
        'Mailed check': 'Mailed Check',
        'Bank transfer (automatic)': 'Bank Transfer',
        'Credit card (automatic)': 'Credit Card'
    })
    return df


def categorical_encoding(df):
    """Encodes categorical columns using one-hot encoding."""
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_data = pd.DataFrame(encoder.fit_transform(df[categorical_cols]), columns=encoder.get_feature_names_out(categorical_cols))
    return encoded_data


def save_data(df, file_path):
    """Saves the processed data to a CSV file."""
    df.to_csv(file_path, index=False)


def main():
    file_path = 'data\\Telco-Customer-Churn.csv'
    df = open_file(file_path)
    df = data_cleaning(df)
    encoded_df = categorical_encoding(df)
    data = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)
    save_data(data, 'data\\processed_data.csv')


if __name__ == '__main__':
    main()