import numpy as np
import pandas as pd

from utils import preparation


# Save the predictions to a CSV file
def save_predictions(predictions):
    predictions_file_path = 'prediction_files/predictions.csv'
    pd.DataFrame(predictions).to_csv(predictions_file_path, index=False,header=False)
    print(f"Predictions saved to {predictions_file_path}")


def get_predictions(data, model):
    predictions = model.predict(data)
    save_predictions(predictions)
    return predictions


# Print the number of companies predicted to go bankrupt
def print_prediction_result(predictions, data):
    number_of_bankruptcies = np.count_nonzero(predictions)
    number_of_companies = data.shape[0]
    print()
    print(number_of_bankruptcies, "out of ", number_of_companies, " companies are predicted to go bankrupt.\n")


def main():
    print()
    data, model = preparation('top10')
    predictions = get_predictions(data, model)
    print_prediction_result(predictions, data)


main()
