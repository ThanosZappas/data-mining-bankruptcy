import numpy as np
import pandas as pd
import joblib  # Used to load the saved model
from typing import Any

Top10_Features_HistGradientBoostingClassifier = 'models/Top10_Features_HistGradientBoostingClassifier.pkl'
top_10_features = ['X27', 'X11', 'X34', 'X46', 'X9', 'X58', 'X5', 'X6', 'X47', 'X13']

Top63_Features_HistGradientBoostingClassifier = 'models/Top63_Features_HistGradientBoostingClassifier.pkl'
# top_63_features = ['X21','X37','X65']

def preparation(features : Any = 'top63'):
    def get_data(features):
        if features == 'top10':
            data = pd.read_csv('training_companydata.csv', na_values=['?'])
            data = data.loc[:, top_10_features]
        else:
            data = pd.read_csv('training_companydata.csv', na_values=['?'])
            data = data.drop(columns=['X21','X37','X65']) # TODO - Add the top 63 features
        return data

    def get_model(features):
        if features == 'top10':
            model = joblib.load(Top10_Features_HistGradientBoostingClassifier)
        else:
            model = joblib.load(Top63_Features_HistGradientBoostingClassifier)
        return model
    return get_data(features), get_model(features)

# Insert 'top10' or 'top63'
data, model = preparation('top10')

# Make predictions
predictions = model.predict(data)

# Print the number of companies predicted to go bankrupt
number_of_bankruptcies = np.count_nonzero(predictions)
number_of_companies = data.shape[0]
print()
print(number_of_bankruptcies, "out of ", number_of_companies, " companies are predicted to go bankrupt.\n")

# Save predictions to a file
predictions_file_path = 'predictions.csv'
pd.DataFrame(predictions).to_csv(predictions_file_path, index=False)
print(f"Predictions saved to {predictions_file_path}")
