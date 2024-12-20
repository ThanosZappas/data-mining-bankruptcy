import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.inspection import permutation_importance
import time

def variable_initialization():
    # Initialize variables to store accuracies and confusion matrices
    accuracies = []
    confusion_matrices = []
    classification_reports = []
    f1_scores = []
    feature_importances_list = []
    return accuracies, confusion_matrices, classification_reports, f1_scores, feature_importances_list

def get_mean_feature_importances(feature_importances_list,X):
    mean_feature_importances = np.mean(feature_importances_list, axis=0)
    return pd.Series(mean_feature_importances, index=X.columns).sort_values(ascending=False)

def train_predict(X, y, model, iterations, train_size, permutation_importance_flag=False):
    accuracies, confusion_matrices, classification_reports, f1_scores, feature_importances_list = variable_initialization()
    for _ in range(iterations):
        
        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)

        # Model Training
        model.fit(X_train, y_train)
        # Make Predictions
        y_pred = model.predict(X_test)

        # Calculate accuracy and confusion matrix on the test set
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        confusion_matrices.append(confusion_matrix(y_test, y_pred))
        
        # Store classification report
        report = classification_report(y_test, y_pred, output_dict=True, target_names=['Not Bankrupt', 'Bankrupt'])
        classification_reports.append(report)

        # Calculate F1-score for the positive class (Bankrupt in this case)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        f1_scores.append(f1)

        if permutation_importance_flag==True:
            # Calculate feature importance
            result = permutation_importance(model, X_test, y_test, n_repeats=10, n_jobs=-1)
            feature_importances_list.append(result.importances_mean)

    return accuracies, confusion_matrices, classification_reports, f1_scores, feature_importances_list
    
def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def get_mean_report(classification_reports):
    mean_report = {}
    for key in classification_reports[0].keys():
        if isinstance(classification_reports[0][key], dict):
            mean_report[key] = {}
            for metric in classification_reports[0][key].keys():
                mean_report[key][metric] = np.mean([report[key][metric] for report in classification_reports if isinstance(report[key], dict)])
        else:
            mean_report[key] = np.mean([report[key] for report in classification_reports])
    return mean_report

def print_results( accuracies, confusion_matrices, classification_reports, f1_scores, start_time, end_time):
    mean_report = get_mean_report(classification_reports)
    print()
    print(f"Mean accuracy: {np.mean(accuracies) * 100:.2f}% with Standard deviation: {np.std(accuracies) * 100:.2f}%\n")
    print("Mean Confusion Matrix:\n", np.mean(confusion_matrices, axis=0), "\n")
    print("Mean Classification Report:\n", pd.DataFrame(mean_report).transpose(), "\n")
    print(f"Mean F1-score for Bankrupt class: {np.mean(f1_scores):.2f}, with Standard Deviation of {np.std(f1_scores):.2f}\n")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print("\n\n")

def print_top_features(feature_importances_list, X, number_of_features):
    mean_feature_importances = get_mean_feature_importances(feature_importances_list,X)
    print(f"Mean Feature Importances:\n{mean_feature_importances.head(number_of_features)}\n")

def main():
    print()

    data = pd.read_csv('data/updated_training_companydata.csv')
    X = data.drop(columns=['X65'])
    y = data['X65']
    model = HistGradientBoostingClassifier(max_depth=10, early_stopping=False)

    start_time = time.time()
    accuracies, confusion_matrices, classification_reports, f1_scores, feature_importances_list = train_predict(X, y, model, iterations=3, train_size= 0.8, permutation_importance_flag=True)
    end_time = time.time()

    # save_model(model, "models/HistGradientBoostingClassifier_Top63_Features.pkl")
    print("Results for 63 features:")
    print_results( accuracies, confusion_matrices, classification_reports, f1_scores,start_time, end_time)
    print_top_features(feature_importances_list, X, 10)

    trimmed_data = data[get_mean_feature_importances(feature_importances_list,X).head(10).index]
    X = trimmed_data
    y = data['X65']


    start_time = time.time()
    accuracies, confusion_matrices, classification_reports, f1_scores, _ = train_predict(X, y, model, iterations=3, train_size= 0.8, permutation_importance_flag=False)
    end_time = time.time()

    # save_model(model, "models/HistGradientBoostingClassifier_Top10_Features.pkl")
    print("Results for top 10 features:")
    print_results( accuracies, confusion_matrices, classification_reports, f1_scores,start_time, end_time)

main()