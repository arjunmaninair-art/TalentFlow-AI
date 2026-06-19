import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

print("⏳ Step 1: Generating applicant screening dataset...")

# Day 1 & 2: Synthetic Data Generation
np.random.seed(42)
num_applicants = 400

# Features: Years of Experience, Skill Match % (0-100), Technical Test Score (0-100)
years_exp = np.random.randint(0, 15, size=num_applicants)
skill_match = np.random.randint(40, 100, size=num_applicants)
test_score = np.random.randint(30, 100, size=num_applicants)

# Target: 1 = Invite to Interview, 0 = Reject
# Logic: High experience, high skill match, and passing test scores trigger an interview
score = (years_exp * 3) + (skill_match * 0.5) + (test_score * 0.6)
interview_status = np.where(score > 65, 1, 0)

df = pd.DataFrame({
    'years_experience': years_exp,
    'skill_match_percentage': skill_match,
    'technical_test_score': test_score,
    'interview_status': interview_status
})

# Day 3: Train Scikit-Learn Model
X = df[['years_experience', 'skill_match_percentage', 'technical_test_score']]
y = df['interview_status']

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)
print("✅ Applicant screening classification model trained!")

# Save using Joblib
joblib.dump(model, 'recruiter_model.joblib')
print("💾 Model successfully frozen as 'recruiter_model.joblib'")