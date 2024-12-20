import pandas as pd
import joblib
from typing import Any

dataset_file = 'data/test_unlabeled.csv'
target_column = ['X65'] # Bankruptcy indicator

HistGradientBoostingClassifier_Top10_Features = 'models/HistGradientBoostingClassifier_Top10_Features.pkl'
top_10_features = ['X27', 'X11', 'X34', 'X46', 'X9', 'X58', 'X5', 'X6', 'X47', 'X13']

HistGradientBoostingClassifier_Top63_Features = 'models/HistGradientBoostingClassifier_Top63_Features.pkl'
unused_features = ['X21','X37','X65']

def preparation(features: Any = 'top63'):
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

def rank_businesses_by_risk(model, X_test):
    probabilities = model.predict_proba(X_test)[:, 1]  # Probability of bankruptcy class
    ranking_df = pd.DataFrame({
        'Bankruptcy_Probability': probabilities
    })
    ranking_df = ranking_df.sort_values(by='Bankruptcy_Probability', ascending=False)
    return ranking_df

def display_top_features(top_n=10):
    print(f"Top {top_n} Features by Importance:\n", top_10_features)

def display_top_risk_businesses(ranking_df, top_n=50):
    print(f"Top {top_n} Businesses with Highest Bankruptcy Risk:")
    print(ranking_df.head(top_n))

def main():

    # Load the model and data
    data, model = preparation('top10')

    # Rank businesses by bankruptcy risk
    ranking_df = rank_businesses_by_risk(model, data)
    display_top_risk_businesses(ranking_df, 50)

main()