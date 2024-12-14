import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

# Load data
data = pd.read_csv('training_companydata.csv', na_values=['?'])
print(data.shape)
print(data.head)

# Drop Columns with High Missing Values
missing_threshold = 0.2
missing_percentage = data.isnull().mean()
to_drop_missing = missing_percentage[missing_percentage > missing_threshold].index
data = data.drop(columns=to_drop_missing)

# Scale the Dataset
# Fill missing values temporarily to scale; use mean or median if needed
# data_filled = data.fillna(data.mean())
data_filled = data.fillna(0)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_filled)
scaled_data = pd.DataFrame(scaled_data, columns=data.columns)


# print(data.shape)
#
# # Drop Columns Based on High Correlation
# # Create an upper triangle matrix of correlations
# correlation_matrix = data.corr().abs()
# upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
#
# # Find index of features with correlation greater than threshold
# threshold = 0.95 # Extra safe threshold
# to_drop_corr = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
# data = data.drop(columns=to_drop_corr)

# print(f"Dropped columns due to high correlation: {to_drop_corr}")
# print(data.shape)

# Drop Low-Variance Columns
# variance_threshold = 0.01
# selector = VarianceThreshold(threshold=variance_threshold)
# selector.fit(data.fillna(0))
# low_variance_columns = [column for column in data.columns if column not in data.columns[selector.get_support()]]
# data = data.drop(columns=low_variance_columns)
# # print(data.shape)

# Final output
print("Remaining columns:", data.columns)
print(data.shape)
print(data.head())
data.to_csv('updated_training_companydata.csv', index=False)


#testing part 2


X = data.drop(columns=['X65'])
y = data['X65']

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 4: Model Training
# Initialize the HistGradientBoostingClassifier
clf = HistGradientBoostingClassifier()
clf.fit(X_train, y_train)

# Step 5: Make Predictions
y_pred = clf.predict(X_test)

# Step 6: Evaluate Model
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 7: Feature Importance
# Calculate permutation importance on the test set
result = permutation_importance(clf, X_test, y_test, n_repeats=10, n_jobs=-1)

# Extract feature importances and sort
feature_importances = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)
print("Top 10 Features by Importance:\n", feature_importances.head(10))
