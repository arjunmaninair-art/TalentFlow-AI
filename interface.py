import streamlit as st
import requests

# Set up page configuration
st.set_page_config(
    page_title="TalentFlow AI Portal",
    page_icon="💼",
    layout="centered"
)

st.title("💼 TalentFlow AI Portal")
st.subheader("Intelligent Screening & Automated Outreach Pipeline")
st.write("Enter the candidate's metrics below to run the Scikit-Learn screening model and generate a custom Groq recruiter email.")

st.divider()

# Form layout for user inputs
with st.form("candidate_form"):
    st.markdown("### 📋 Candidate Profile & Performance Metrics")
    
    col1, col2 = st.columns(2)
    with col1:
        candidate_name = st.text_input("Candidate Name", placeholder="e.g., Arjun Kumar")
    with col2:
        core_skill = st.text_input("Core Specialization", placeholder="e.g., Machine Learning")
        
    st.divider()
    st.markdown("### 👤 Recruiter Information")
    recruiter_name = st.text_input("Your Name (For Email Sign-off)", value="Arjun")
    
    st.divider()
    
    years_experience = st.slider("Years of Experience", min_value=0, max_value=15, value=3)
    skill_match_percentage = st.slider("Job Skill Match (%)", min_value=0, max_value=100, value=75)
    technical_test_score = st.slider("Technical Assessment Score", min_value=0, max_value=100, value=70)
    
    # INDENTATION FIXED: This button must be indented so it sits inside the "with" block
    submit_button = st.form_submit_button(label="🚀 Screen Candidate & Draft Email")

# Execution logic sits outside the form block (No indentation)
if submit_button:
    if not candidate_name or not core_skill or not recruiter_name:
        st.error("⚠️ Please fill out all text fields (Candidate Name, Specialization, and Recruiter Name).")
    else:
        payload = {
            "candidate_name": candidate_name,
            "core_skill": core_skill,
            "years_experience": float(years_experience),
            "skill_match_percentage": float(skill_match_percentage),
            "technical_test_score": float(technical_test_score),
            "recruiter_name": recruiter_name
        }
        
        # Pointing to your active backend port
        backend_url = "http://127.0.0.1:8080/screen_candidate"
        
        with st.spinner("Processing metrics through Scikit-Learn and generating email via Groq..."):
            try:
                response = requests.post(backend_url, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("✅ Execution Successful!")
                    
                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.metric(label="Candidate Name", value=result["candidate_profile"]["name"])
                    with res_col2:
                        decision = result["screening_results"]["ml_decision"]
                        st.metric(label="ML Pipeline Decision", value=decision)
                    
                    st.divider()
                    st.markdown("### ✉️ Automated Recruiter Outreach Email (Groq GenAI)")
                    st.text_area(
                        label="Copy/Paste Template:",
                        value=result["automated_outreach_email"],
                        height=350
                    )
                else:
                    st.error(f"❌ Backend Error (Status Code: {response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"❌ Failed to connect to the FastAPI backend server. Is it running? Details: {e}")