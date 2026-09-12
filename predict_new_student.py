import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# --- Rebuild the trained model (same steps as train_model.py) ---
df = pd.read_csv("student_data.csv")
X = df.drop("performs_well", axis=1)
y = df["performs_well"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# --- Predict for a new student ---
def predict_student(attendance, study_hours, previous_marks, assignment_scores, internal_scores):
    new_student = pd.DataFrame([{
        "attendance": attendance,
        "study_hours": study_hours,
        "previous_marks": previous_marks,
        "assignment_scores": assignment_scores,
        "internal_scores": internal_scores
    }])
    new_student_scaled = scaler.transform(new_student)
    prediction = model.predict(new_student_scaled)[0]
    probability = model.predict_proba(new_student_scaled)[0][1]  # chance of "performs well"

    result = "likely to perform WELL" if prediction == 1 else "AT RISK of underperforming"
    print(f"\nPrediction: {result}")
    print(f"Confidence: {probability*100:.1f}% chance of performing well")
    return prediction

# --- Try it out with a sample student ---
print("Example 1: Strong student profile")
predict_student(attendance=92, study_hours=6, previous_marks=85, assignment_scores=88, internal_scores=80)

print("\nExample 2: At-risk student profile")
predict_student(attendance=55, study_hours=1, previous_marks=45, assignment_scores=40, internal_scores=38)

# --- Try your own! Edit the numbers below and run again ---
print("\nExample 3: Your own test case")
predict_student(attendance=70, study_hours=3, previous_marks=60, assignment_scores=65, internal_scores=55)