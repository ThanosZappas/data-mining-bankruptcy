import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

# Load data
data = pd.read_csv('training_companydata.csv', na_values=['?'])
print(data.shape)

# Scale the Dataset
# Fill missing values temporarily to scale; use mean or median if needed
data_filled = data.fillna(data.mean())
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_filled)
scaled_data = pd.DataFrame(scaled_data, columns=data.columns)


# Drop Columns with High Missing Values
missing_threshold = 0.2
missing_percentage = data.isnull().mean()
to_drop_missing = missing_percentage[missing_percentage > missing_threshold].index
data = data.drop(columns=to_drop_missing)
# print(data.shape)

# Drop Columns Based on High Correlation
# Create an upper triangle matrix of correlations
correlation_matrix = data.corr().abs()
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

# Find index of features with correlation greater than threshold
threshold = 0.95 # Extra safe threshold
to_drop_corr = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
data = data.drop(columns=to_drop_corr)

# print(f"Dropped columns due to high correlation: {to_drop_corr}")
# print(data.shape)

# Drop Low-Variance Columns
variance_threshold = 0.01
selector = VarianceThreshold(threshold=variance_threshold)
selector.fit(data.fillna(0))
low_variance_columns = [column for column in data.columns if column not in data.columns[selector.get_support()]]
data = data.drop(columns=low_variance_columns)
# print(data.shape)

# Final output
print("Remaining columns:", data.columns)
print(data.shape)
print(data.head())
data.to_csv('updated_training_companydata.csv', index=True)
