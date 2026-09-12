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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

# Train the model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Predict on the unseen test set
y_pred = model.predict(X_test_scaled)

# Evaluate
print("\n--- Model Performance ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nFull report:")
print(classification_report(y_test, y_pred))

# Which features matter most to the model?
print("\nFeature importance (coefficients):")
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"{feature}: {coef:.3f}")