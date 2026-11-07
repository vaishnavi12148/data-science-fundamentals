"""
Data Science Fundamentals Assessment
Demonstrates Python basics, data types, statistics, data manipulation,
visualization, and responsible data-science practices.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Create a small sample dataset
data = {
    "Student": ["Asha", "Ravi", "Meera", "Kabir", "Neha", "Arjun"],
    "Hours_Studied": [2, 3, 4, 5, 6, 7],
    "Attendance": [70, 75, 80, 85, 90, 95],
    "Score": [48, 55, 65, 72, 84, 91],
    "Department": ["IT", "CS", "IT", "CS", "IT", "CS"],
}
df = pd.DataFrame(data)

print("DATASET")
print(df.to_string(index=False))

# 2. Inspect data
print("\nSHAPE:", df.shape)
print("\nDATA TYPES:\n", df.dtypes)
print("\nMISSING VALUES:\n", df.isnull().sum())
print("\nSUMMARY STATISTICS:\n", df[["Hours_Studied", "Attendance", "Score"]].describe())

# 3. Basic Python variables, operators, conditions, and loops
target_score = 60
passed = df["Score"] >= target_score
print("\nPASS STATUS:\n", pd.DataFrame({"Student": df["Student"], "Passed": passed}))

# 4. Data manipulation: filtering, sorting, grouping
high_scorers = df[df["Score"] >= 75].sort_values("Score", ascending=False)
print("\nSTUDENTS SCORING 75 OR ABOVE:\n", high_scorers.to_string(index=False))

department_summary = df.groupby("Department")["Score"].agg(
    ["count", "mean", "min", "max"]
)
print("\nDEPARTMENT SUMMARY:\n", department_summary)

# 5. Statistical foundations
scores = df["Score"]
print("\nMEAN:", scores.mean())
print("MEDIAN:", scores.median())
print("MODE:", scores.mode().tolist())
print("RANGE:", scores.max() - scores.min())
print("POPULATION VARIANCE:", scores.var(ddof=0))
print("POPULATION STANDARD DEVIATION:", scores.std(ddof=0))
print("SCORE-HOURS CORRELATION:", df["Score"].corr(df["Hours_Studied"]))

# 6. Visualization
plt.figure(figsize=(7, 4))
plt.bar(df["Student"], df["Score"])
plt.title("Student Scores")
plt.xlabel("Student")
plt.ylabel("Score")
plt.tight_layout()
plt.savefig("student_scores.png", dpi=150)
plt.show()

plt.figure(figsize=(7, 4))
plt.scatter(df["Hours_Studied"], df["Score"])
plt.title("Study Hours vs Score")
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.tight_layout()
plt.savefig("study_hours_vs_score.png", dpi=150)
plt.show()

# 7. A simple data-quality check
assert df["Student"].is_unique, "Student names should be unique in this example."
assert df["Score"].between(0, 100).all(), "Scores must be between 0 and 100."
print("\nData-quality checks passed.")
