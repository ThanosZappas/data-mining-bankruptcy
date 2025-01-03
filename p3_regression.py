import pandas as pd

from utils import preparation


def rank_businesses_by_risk(model, X_test):
    probabilities = model.predict_proba(X_test)[:, 1]  # Probability of bankruptcy class
    ranking_df = pd.DataFrame({
        'Company_ID': X_test.index,
        'Bankruptcy_Probability': probabilities
    })
    ranking_df = ranking_df.sort_values(by='Bankruptcy_Probability', ascending=False)
    return ranking_df


def display_top_risk_businesses(ranking_df, top_n=50):
    print(f"Top {top_n} Businesses with Highest Bankruptcy Risk:")
    print(ranking_df.head(top_n).to_string(index=False))


def save_predictions(ranking_df):
    predictions_file_path = 'prediction_files/ranking.csv'
    pd.DataFrame(ranking_df.head(50)).to_csv(predictions_file_path, sep='\t', index=False)
    print(f"Predictions saved to {predictions_file_path}")


def main():
    # Load the model and data
    data, model = preparation('top10')

    # Rank businesses by bankruptcy risk
    ranking_df = rank_businesses_by_risk(model, data)
    display_top_risk_businesses(ranking_df, 50)
    save_predictions(ranking_df)


main()
