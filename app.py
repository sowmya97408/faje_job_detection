import streamlit as st
import re
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib"])
import joblib

# Load trained machine learning files
model = joblib.load("fake_job_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Clean job description
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Website design
st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Fake Job Detection")
st.subheader("Using Machine Learning")

st.write(
    "Enter a job description below to check whether "
    "the job posting is potentially genuine or fraudulent."
)

job_description = st.text_area(
    "Enter Job Description",
    height=250,
    placeholder="Paste the complete job description here..."
)

if st.button("Check Job Posting"):
    if not job_description.strip():
        st.warning("Please enter a job description.")
    else:
        cleaned = clean_text(job_description)
        transformed = tfidf.transform([cleaned])
        prediction = model.predict(transformed)[0]

        if prediction == 1:
            st.error("⚠️ Potentially FRAUDULENT Job Posting")
        else:
            st.success("✅ Likely GENUINE Job Posting")

st.info(
    "Note: This prediction is only a screening aid. "
    "Always verify the employer independently."
)
