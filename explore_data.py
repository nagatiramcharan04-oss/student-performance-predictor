import pandas as pd

# Load the dataset
df = pd.read_csv("student_data.csv")

# 1. Basic shape and structure
print("Shape (rows, columns):", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# 2. Summary statistics — mean, std, min, max for each column
print("\nSummary statistics:")
print(df.describe())

# 3. Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# 4. How balanced is our target? (important — if 95% are "1", accuracy is a misleading metric)
print("\nTarget distribution:")
print(df['performs_well'].value_counts())
print(df['performs_well'].value_counts(normalize=True))  # as percentages

# 5. Correlation with the target — which features matter most?
print("\nCorrelation with performs_well:")
print(df.corr()['performs_well'].sort_values(ascending=False))