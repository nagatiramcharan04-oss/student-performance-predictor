import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# --- Load data and train the model (runs once when the app starts) ---
@st.cache_data
def load_and_train():
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

    return model, scaler, X.columns

model, scaler, feature_names = load_and_train()

# --- Page setup ---
st.title("🎓 AI Student Performance Predictor")
st.write("Enter a student's stats below to predict whether they're likely to perform well.")

# --- Input widgets (this replaces manually editing Python code!) ---
attendance = st.slider("Attendance (%)", 0, 100, 80)
study_hours = st.slider("Study hours per day", 0.0, 12.0, 4.0)
previous_marks = st.slider("Previous marks", 0, 100, 65)
assignment_scores = st.slider("Assignment scores", 0, 100, 70)
internal_scores = st.slider("Internal scores", 0, 100, 68)

# --- Predict button ---
if st.button("Predict"):
    new_student = pd.DataFrame([{
        "attendance": attendance,
        "study_hours": study_hours,
        "previous_marks": previous_marks,
        "assignment_scores": assignment_scores,
        "internal_scores": internal_scores
    }])
    new_student_scaled = scaler.transform(new_student)
    prediction = model.predict(new_student_scaled)[0]
    probability = model.predict_proba(new_student_scaled)[0][1]

    if prediction == 1:
        st.success(f"✅ Likely to perform WELL — {probability*100:.1f}% confidence")
    else:
        st.error(f"⚠️ AT RISK of underperforming — {(1-probability)*100:.1f}% confidence")