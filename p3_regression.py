import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

# Load Data
data = pd.read_csv('updated_training_companydata.csv')

# Define features and target
X = data.drop(columns=['X65'])  # Replace 'X65' with the actual target column name if needed
y = data['X65']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

# Model Training
clf = HistGradientBoostingClassifier()
clf.fit(X_train, y_train)

# Feature Importance
result = permutation_importance(clf, X_test, y_test, n_repeats=10, n_jobs=-1)
feature_importances = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)
top_10_features = feature_importances.head(10).index
print("Top 10 Features by Importance:\n", feature_importances.head(10))

# # Trim dataset to top 10 features
# X_top_10_features = X[top_10_features]
#
# # Train-Test Split for trimmed dataset
# X_train, X_test, y_train, y_test = train_test_split(X_top_10_features, y, train_size=0.7)
#
# # Re-train the model with top 10 features
# clf.fit(X_train, y_train)

# Rank businesses by predicted bankruptcy probability
probabilities = clf.predict_proba(X_test)[:, 1]  # Probability of bankruptcy class

# Create a ranking DataFrame
ranking_df = pd.DataFrame({
    'Bankruptcy_Probability': probabilities
})

# Sort by bankruptcy probability in descending order
ranking_df = ranking_df.sort_values(by='Bankruptcy_Probability', ascending=False)

# Display top 10 businesses with the highest risk
print("Top 50 Businesses with Highest Bankruptcy Risk:")
print(ranking_df.head(100))
