import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

# Step 2: Load Data

# #default dataset
# data = pd.read_csv('training_companydata.csv',na_values=['?'])
# # Scale the Dataset
# # Fill missing values temporarily to scale; use mean or median if needed
# data_filled = data.fillna(data.mean())
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(data_filled)
# scaled_data = pd.DataFrame(scaled_data, columns=data.columns)

# Processed Dataset
data = pd.read_csv('updated_training_companydata.csv')

X = data.drop(columns=['X65'])
y = data['X65']
print(data.shape)
# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Training
# Initialize the HistGradientBoostingClassifier
clf = HistGradientBoostingClassifier(random_state=42)
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
result = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)

# Extract feature importance's and sort
feature_importances = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)
print("Top 10 Features by Importance:\n", feature_importances.head(10))