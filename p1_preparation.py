import pandas as pd
from sklearn.preprocessing import StandardScaler


# Drop Columns with High Missing Values
def drop_columns(data):
    missing_threshold = 0.2
    missing_percentage = data.isnull().mean()
    to_drop_missing = missing_percentage[missing_percentage > missing_threshold].index
    data = data.drop(columns=to_drop_missing)
    return data


# Scale the Data
def scale_data(data):
    target_column = 'X65'
    target_data = data[target_column]
    features_data = data.drop(columns=[target_column])

    # Fill missing values temporarily to scale; use mean or median if needed
    features_filled = features_data.fillna(0)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_filled)

    scaled_features = pd.DataFrame(scaled_features, columns=features_data.columns)

    scaled_data = pd.concat([scaled_features, target_data.reset_index(drop=True)], axis=1)
    return scaled_data


# Preprocess Data
def preprocess_data(data):
    data = drop_columns(data)
    data = scale_data(data)
    return data


def main():
    # Load data
    data = pd.read_csv('data/training_companydata.csv', na_values=['?'])
    print()
    print(data.shape)
    print("Initial columns: ", data.columns)

    target_column = 'X65'
    data = preprocess_data(data)

    print()
    print(data.shape)
    print("Remaining columns: ", data.columns)
    data.to_csv('data/updated_training_companydata.csv', index=False)


main()
