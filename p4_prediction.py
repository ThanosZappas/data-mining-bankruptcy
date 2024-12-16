import numpy as np
import pandas as pd
import joblib
from typing import Any

HistGradientBoostingClassifier_Top10_Features = 'models/HistGradientBoostingClassifier_Top10_Features.pkl'
top_10_features = ['X27', 'X11', 'X34', 'X46', 'X9', 'X58', 'X5', 'X6', 'X47', 'X13']

HistGradientBoostingClassifier_Top63_Features = 'models/HistGradientBoostingClassifier_Top63_Features.pkl'
unused_features = ['X21','X37','X65']

dataset_file = 'training_companydata.csv'

# Save the predictions to a CSV file
def save_predictions(predictions):
    predictions_file_path = 'predictions.csv'
    pd.DataFrame(predictions).to_csv(predictions_file_path, index=False)
    print(f"Predictions saved to {predictions_file_path}")

def get_predictions(data, model):
    data, model
    predictions = model.predict(data)
    save_predictions(predictions)
    return predictions

# Prepare the data and model ( top10 or top63 features)
def preparation(features : Any = 'top63'):
    def get_data(features):
        if features == 'top10':
            data = pd.read_csv(dataset_file, na_values=['?'])
            data = data.loc[:, top_10_features]
        else:
            data = pd.read_csv(dataset_file, na_values=['?'])
            data = data.drop(columns=unused_features)
        return data

    def get_model(features):
        if features == 'top10':
            model = joblib.load(HistGradientBoostingClassifier_Top10_Features)
        else:
            model = joblib.load(HistGradientBoostingClassifier_Top63_Features)
        return model
    return get_data(features), get_model(features)

# Print the number of companies predicted to go bankrupt
def print_prediction_result(predictions, data):
    number_of_bankruptcies = np.count_nonzero(predictions)
    number_of_companies = data.shape[0]
    print()
    print(number_of_bankruptcies, "out of ", number_of_companies, " companies are predicted to go bankrupt.\n")

def main():
    data, model = preparation('top10')
    predictions = get_predictions(data, model)
    print_prediction_result(predictions, data)

main()