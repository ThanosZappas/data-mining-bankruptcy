import pandas as pd
from sklearn.preprocessing import StandardScaler

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

def main():
    # Load data
    data = pd.read_csv('training_companydata.csv', na_values=['?'])
    print()
    print(data.shape)
    print("Initial columns: ", data.columns)
    data = preprocess_data(data)
    print()
    print(data.shape)
    print("Remaining columns: ", data.columns)
    data.to_csv('updated_training_companydata.csv', index=False)

main()