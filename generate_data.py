import numpy as np
import pandas as pd

np.random.seed(42)  # keeps results reproducible

n_students = 500

# Generate base features
attendance = np.random.normal(80, 12, n_students).clip(40, 100)
study_hours = np.random.normal(4, 2, n_students).clip(0, 12)
previous_marks = np.random.normal(65, 15, n_students).clip(20, 100)
assignment_scores = np.random.normal(70, 15, n_students).clip(0, 100)
internal_scores = np.random.normal(68, 14, n_students).clip(0, 100)

# Combine features into a weighted "performance score" with some randomness
performance_score = (
    0.25 * attendance +
    0.20 * (study_hours * 8) +   # scale study hours up to a comparable range
    0.25 * previous_marks +
    0.15 * assignment_scores +
    0.15 * internal_scores +
    np.random.normal(0, 5, n_students)  # noise, so it's not a perfect formula
)

# Label: "1" = likely to perform well, "0" = at risk
threshold = np.percentile(performance_score, 40)  # bottom 40% = at risk
performs_well = (performance_score > threshold).astype(int)

df = pd.DataFrame({
    "attendance": attendance.round(1),
    "study_hours": study_hours.round(1),
    "previous_marks": previous_marks.round(1),
    "assignment_scores": assignment_scores.round(1),
    "internal_scores": internal_scores.round(1),
    "performs_well": performs_well
})

df.to_csv("student_data.csv", index=False)
print(df.head())
print(f"\nGenerated {n_students} students. {df['performs_well'].sum()} labeled as performing well.")