import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from groq import Groq
from dotenv import load_dotenv

# 1. Initialize FastAPI instance
app = FastAPI(
    title="Automated Job Applicant Screening API",
    description="Uses Scikit-Learn to screen candidates and Groq LPU to draft recruiter outreach emails."
)

load_dotenv()

# 2. Initialize Groq Client safely using environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY environment variable is missing!")

groq_client = Groq(api_key=GROQ_API_KEY)
# 3. Load the saved Scikit-Learn model globally
try:
    model = joblib.load('recruiter_model.joblib')
    print("🎯 Scikit-Learn validation model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model. Did you run train_model.py first? Details: {e}")

# 4. Define the incoming payload structure
class CandidatePayload(BaseModel):
    candidate_name: str
    core_skill: str
    years_experience: float
    skill_match_percentage: float
    technical_test_score: float
    recruiter_name: str

# Helper function to trigger Groq inference
def generate_recruiter_email(candidate_name: str, core_skill: str, prediction: int, recruiter_name: str) -> str:
    system_prompt = (
        f"You are an elite, professional corporate recruiter named {recruiter_name}. Your job is to draft clear, personalized, "
        "and impactful emails to job applicants based on their application results. "
        f"Always sign off the email using your name ({recruiter_name}). Do not use generic placeholders like [Your Name]."
    )
    
    if prediction == 1:
        user_prompt = (
            f"Draft a warm, professional 'Interview Invitation' email to {candidate_name}. "
            f"Mention that we were highly impressed by their background in {core_skill}. "
            f"Ask them to reply with their availability for a technical interview round next week. "
            f"Sign off explicitly as {recruiter_name}, Corporate Recruiter."
        )
    else:
        user_prompt = (
            f"Draft a polite, encouraging 'Rejection' email to {candidate_name}. "
            f"They applied with a background in {core_skill}. Thank them for their time, "
            f"let them know we chose to proceed with other candidates whose profiles matched closer at this time, "
            f"and wish them luck. Sign off explicitly as {recruiter_name}, Corporate Recruiter."
        )

    try:
        completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.6
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error connecting to Groq LPU engine: {str(e)}"

# 5. API Endpoint setup
@app.post("/screen_candidate")
async def screen_candidate(payload: CandidatePayload):
    try:
        years_exp = payload.years_experience
        skill_match = payload.skill_match_percentage
        test_score = payload.technical_test_score

        # Format feature array for Scikit-Learn inference
        features = np.array([[years_exp, skill_match, test_score]])
        prediction = int(model.predict(features)[0])
        
        status_string = "Proceed to Interview" if prediction == 1 else "Reject/Hold"

        # Generate the dynamic email using Groq
        email_body = generate_recruiter_email(
            payload.candidate_name, 
            payload.core_skill, 
            prediction, 
            payload.recruiter_name
        )

        return {
            "status": "success",
            "candidate_profile": {
                "name": payload.candidate_name,
                "specialization": payload.core_skill
            },
            "screening_results": {
                "ml_decision": status_string,
                "internal_code": prediction
            },
            "automated_outreach_email": email_body
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))