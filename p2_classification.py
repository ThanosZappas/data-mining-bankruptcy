import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.inspection import permutation_importance
import time

# Load Data
data = pd.read_csv('updated_training_companydata.csv')

X = data.drop(columns=['X65'])
y = data['X65']

# Initialize variables to store accuracies and confusion matrices
accuracies = []
confusion_matrices = []
classification_reports = []
f1_scores = []
feature_importances_list = []
clf = HistGradientBoostingClassifier(max_depth=10, early_stopping=False)



# Start the timer
start_time = time.time()

# Perform 10 iterations
for _ in range(100):
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    # Model Training
    clf.fit(X_train, y_train)

    # Make Predictions
    y_pred = clf.predict(X_test)

    # Calculate accuracy and confusion matrix on the test set
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    confusion_matrices.append(confusion_matrix(y_test, y_pred))
    # Store classification report
    report = classification_report(y_test, y_pred, output_dict=True, target_names=['Not Bankrupt', 'Bankrupt'])
    classification_reports.append(report)

    # Calculate F1-score for the positive class (Bankrupt in this case)
    f1 = f1_score(y_test, y_pred, pos_label=1)  # Adjust `pos_label` as needed
    f1_scores.append(f1)

    # Calculate feature importance
    result = permutation_importance(clf, X_test, y_test, n_repeats=10, n_jobs=-1)
    feature_importances_list.append(result.importances_mean)

# End the timer
end_time = time.time()

# Save the trained model to a file
# model_filename = "models/Top63_Features_HistGradientBoostingClassifier.pkl"
# joblib.dump(clf, model_filename)

# Calculate mean and standard deviation of accuracies
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

# Calculate mean confusion matrix
mean_confusion_matrix = np.mean(confusion_matrices, axis=0)

# Calculate mean classification report
mean_report = {}
for key in classification_reports[0].keys():
    if isinstance(classification_reports[0][key], dict):
        mean_report[key] = {}
        for metric in classification_reports[0][key].keys():
            mean_report[key][metric] = np.mean([report[key][metric] for report in classification_reports if isinstance(report[key], dict)])
    else:
        mean_report[key] = np.mean([report[key] for report in classification_reports])


# Calculate total execution time
execution_time = end_time - start_time

# Print results
print(data.shape, "\n")
print(f"Mean accuracy: {mean_accuracy * 100:.2f}% with Standard deviation: {std_accuracy * 100:.2f}%\n")
# print(f"Standard deviation of accuracy: {std_accuracy * 100:.2f}%")
print("Mean Confusion Matrix:\n", mean_confusion_matrix, "\n")
print("Mean Classification Report:\n", pd.DataFrame(mean_report).transpose(), "\n")

# Calculate mean F1-score for the positive class (Bankrupt in this case)
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)
print(f"Mean F1-score for Bankrupt class: {mean_f1:.2f}, with Standard Deviation of {std_f1:.2f}\n")
# print(f"Standard deviation of F1-score for Bankrupt class: +/-{std_f1:.2f}")
print(f"Total execution time: {execution_time:.2f} seconds")

print("\n\n")

# Calculate mean feature importances
mean_feature_importances = np.mean(feature_importances_list, axis=0)
top_features = pd.Series(mean_feature_importances, index=X.columns).sort_values(ascending=False)

# # Feature Importance
# result = permutation_importance(clf, X_test, y_test, n_repeats=10, n_jobs=-1)
# feature_importances = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)

print("Top 10 Features by Importance:\n", top_features.head(10))

print("\n\n")


# Same model with trimmed dataset (top 10 features)

trimmed_data = data[top_features.head(10).index]

X = trimmed_data
y = data['X65']

# Initialize variables to store accuracies and confusion matrices for trimmed data
accuracies = []
confusion_matrices = []

# Start the timer for trimmed data
start_time = time.time()

# Perform 10 iterations for trimmed data
for _ in range(100):
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    # Model Training
    clf.fit(X_train, y_train)

    # Make Predictions
    y_pred = clf.predict(X_test)

    # Calculate accuracy and confusion matrix on the test set
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    confusion_matrices.append(confusion_matrix(y_test, y_pred))
    # Store classification report
    report = classification_report(y_test, y_pred, output_dict=True, target_names=['Not Bankrupt', 'Bankrupt'])
    classification_reports.append(report)

    # Calculate F1-score for the positive class (Bankrupt in this case)
    f1 = f1_score(y_test, y_pred, pos_label=1)  # Adjust `pos_label` as needed
    f1_scores.append(f1)

# End the timer for trimmed data
end_time = time.time()

# Calculate mean and standard deviation of accuracies for trimmed data
mean_accuracy = np.mean(accuracies)
std_accuracy= np.std(accuracies)

# Calculate mean confusion matrix for trimmed data
mean_confusion_matrix = np.mean(confusion_matrices, axis=0)

# Calculate mean classification report
mean_report = {}
for key in classification_reports[0].keys():
    if isinstance(classification_reports[0][key], dict):
        mean_report[key] = {}
        for metric in classification_reports[0][key].keys():
            mean_report[key][metric] = np.mean([report[key][metric] for report in classification_reports if isinstance(report[key], dict)])
    else:
        mean_report[key] = np.mean([report[key] for report in classification_reports])


# Calculate total execution time for trimmed data
execution_time = end_time - start_time

# Print results for trimmed data
print("Results with top 10 features:")
print(trimmed_data.shape, "\n")
print(f"Mean accuracy: {mean_accuracy * 100:.2f}% with Standard deviation: {std_accuracy * 100:.2f}%\n")
# print(f"Standard deviation of accuracy: {std_accuracy * 100:.2f}%")
print("Mean Confusion Matrix:\n", mean_confusion_matrix, "\n")
print("Mean Classification Report:\n", pd.DataFrame(mean_report).transpose(), "\n")

# Calculate mean F1-score for the positive class (Bankrupt in this case)
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)
print(f"Mean F1-score for Bankrupt class: {mean_f1:.2f}, with Standard Deviation of {std_f1:.2f}\n")
# print(f"Standard deviation of F1-score for Bankrupt class: +/-{std_f1:.2f}")
print(f"Total execution time: {execution_time:.2f} seconds")

# Save the trained model to a file
model_filename = "models/Top10_Features_HistGradientBoostingClassifier.pkl"
joblib.dump(clf, model_filename)