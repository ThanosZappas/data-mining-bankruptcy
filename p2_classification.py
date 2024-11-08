import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
import time

# Load Data
data = pd.read_csv('updated_training_companydata.csv')

X = data.drop(columns=['X65'])
y = data['X65']
print(data.shape)

# Initialize variables to store accuracies and confusion matrices
accuracies = []
confusion_matrices = []
clf = HistGradientBoostingClassifier()

# Start the timer
start_time = time.time()

# Perform 10 iterations
for _ in range(10):
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

    # Model Training
    clf.fit(X_train, y_train)

    # Make Predictions
    y_pred = clf.predict(X_test)

    # Calculate accuracy and confusion matrix on the test set
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    confusion_matrices.append(confusion_matrix(y_test, y_pred))

# End the timer
end_time = time.time()

# Calculate mean and standard deviation of accuracies
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

# Calculate mean confusion matrix
mean_confusion_matrix = np.mean(confusion_matrices, axis=0)

# Calculate total execution time
execution_time = end_time - start_time

# Print results
print(f"Mean accuracy: {mean_accuracy * 100:.2f}%")
print(f"Standard deviation of accuracy: {std_accuracy * 100:.2f}%")
print("Mean Confusion Matrix:\n", mean_confusion_matrix)
print(f"Total execution time: {execution_time:.2f} seconds")
print()

# Feature Importance
result = permutation_importance(clf, X_test, y_test, n_repeats=10, n_jobs=-1)
feature_importances = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)
print("Top 10 Features by Importance:\n", feature_importances.head(10))
print()

# Same model with trimmed dataset (top 10 features)
top_10_features = feature_importances.head(10).index
trimmed_data = data[top_10_features]

X = trimmed_data
y = data['X65']
print(trimmed_data.shape)

# Initialize variables to store accuracies and confusion matrices for trimmed data
accuracies = []
confusion_matrices = []

# Start the timer for trimmed data
start_time = time.time()

# Perform 10 iterations for trimmed data
for _ in range(10):
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

    # Model Training
    clf.fit(X_train, y_train)

    # Make Predictions
    y_pred = clf.predict(X_test)

    # Calculate accuracy and confusion matrix on the test set
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    confusion_matrices.append(confusion_matrix(y_test, y_pred))

# End the timer for trimmed data
end_time = time.time()

# Calculate mean and standard deviation of accuracies for trimmed data
mean_accuracy = np.mean(accuracies)
std_accuracy= np.std(accuracies)

# Calculate mean confusion matrix for trimmed data
mean_confusion_matrix = np.mean(confusion_matrices, axis=0)

# Calculate total execution time for trimmed data
execution_time = end_time - start_time

# Print results for trimmed data
print(f"Mean accuracy (trimmed data): {mean_accuracy * 100:.2f}%")
print(f"Standard deviation of accuracy (trimmed data): {std_accuracy * 100:.2f}%")
print("Mean Confusion Matrix (trimmed data):\n", mean_confusion_matrix)
print(f"Total execution time (trimmed data): {execution_time:.2f} seconds")