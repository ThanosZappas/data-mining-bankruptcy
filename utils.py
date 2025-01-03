from typing import Any

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

HistGradientBoostingClassifier_Top10_Features = 'models/HistGradientBoostingClassifier_Top10_Features.pkl'
top_10_features = ['X27', 'X11', 'X34', 'X46', 'X9', 'X58', 'X5', 'X6', 'X47', 'X13']
column_names = [f'X{i}' for i in range(1, 65)]

HistGradientBoostingClassifier_Top63_Features = 'models/HistGradientBoostingClassifier_Top63_Features.pkl'
unused_features = ['X21']

dataset_file = 'data/test_unlabeled.csv'


def display_top_features(top_n=10):
    print(f"Top {top_n} Features by Importance:\n", top_10_features)


# Drop Columns with High Missing Values
def drop_columns(data):
    missing_threshold = 0.2
    missing_percentage = data.isnull().mean()
    to_drop_missing = missing_percentage[missing_percentage > missing_threshold].index
    data = data.drop(columns=to_drop_missing)
    return data


# Scale the Dataset
def scale_data(data):
    # Fill missing values temporarily to scale; use mean or median if needed
    data_filled = data.fillna(0)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_filled)
    scaled_data = pd.DataFrame(scaled_data, columns=data.columns)
    return scaled_data


# Preprocess Data
def preprocess_data(data):
    data = drop_columns(data)
    data = scale_data(data)
    return data


# Prepare the data and model ( top10 or top63 features)
def preparation(features: Any = 'top63'):
    def get_data(features):
        if features == 'top10':
            data = pd.read_csv(dataset_file, na_values=['?'], header=None, names=column_names)
            data = preprocess_data(data)
            data = data.loc[:, top_10_features]
        else:
            data = pd.read_csv(dataset_file, na_values=['?'], header=None, names=column_names)
            data = data.rename(columns={'X62': 'X66'})  # training dataset has X66 instead of X62
            data = preprocess_data(data)
            data = data.drop(columns=unused_features)
        return data

    def get_model(features):
        if features == 'top10':
            model = joblib.load(HistGradientBoostingClassifier_Top10_Features)
        else:
            model = joblib.load(HistGradientBoostingClassifier_Top63_Features)
        return model

    return get_data(features), get_model(features)
