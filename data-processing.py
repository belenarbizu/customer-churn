import pandas as pd


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
    return df


def main():
    file_path = 'data\\Telco-Customer-Churn.csv'
    df = open_file(file_path)
    df = data_cleaning(df)
    print("\nData after converting columns:")
    print(df.info())


if __name__ == '__main__':
    main()