import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("student_data.csv")

# Separate features (X) from the target (y)
X = df.drop("performs_well", axis=1)   # everything except the label
y = df["performs_well"]                 # just the label

# Split: 80% for training, 20% held back for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)
print("\nTraining set target balance:")
print(y_train.value_counts(normalize=True))
print("\nTest set target balance:")
print(y_test.value_counts(normalize=True))

# Scale features so they're on a comparable range
# (e.g. study_hours is 0-12 but internal_scores is 0-100 — scaling puts them on equal footing)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # note: transform only, not fit_transform — explained below

print("\nFirst row before scaling:", X_train.iloc[0].values)
print("First row after scaling:", X_train_scaled[0])