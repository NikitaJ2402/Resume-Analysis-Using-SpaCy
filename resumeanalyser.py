import streamlit as st
import spacy
import fitz  # PyMuPDF for PDF processing
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from spacy.matcher import Matcher
import re
import base64

# Load spaCy model (replace with custom model if trained)
nlp = spacy.load("en_core_web_sm")

# Utility functions
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def preprocess_text(text):
    # Remove headers, footers, or other non-content
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_entities(doc):
    entities = {"name": None, "contact": None, "skills": [], "education": [], "experience": []}
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            entities["name"] = ent.text
        elif ent.label_ in ["EMAIL", "PHONE"]:
            entities["contact"] = entities.get("contact", "") + " " + ent.text
        elif ent.label_ == "ORG" or ent.label_ == "EDUCATION":
            entities["education"].append(ent.text)
        elif ent.label_ == "SKILL":
            entities["skills"].append(ent.text)
        elif ent.label_ == "DATE":
            entities["experience"].append(ent.text)
    return entities

def match_skills(text, skill_keywords):
    matcher = Matcher(nlp.vocab)
    skill_patterns = [[{"LOWER": skill.lower()}] for skill in skill_keywords]
    matcher.add("SKILLS", skill_patterns)
    doc = nlp(text)
    matches = matcher(doc)
    skills_found = [doc[start:end].text for match_id, start, end in matches]
    return list(set(skills_found))

def calculate_similarity(resume_text, job_description):
    # Use count vectorizer for basic cosine similarity
    cv = CountVectorizer().fit_transform([resume_text, job_description])
    similarity_matrix = cosine_similarity(cv)
    return similarity_matrix[0][1]

# Streamlit UI
st.title("Resume Analysis and Scoring Tool")
st.write("Upload a resume PDF and analyze it based on predefined skills and job description.")

# Input Fields
uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
job_description = st.text_area("Paste the Job Description here")

# Load predefined skills (could be expanded with domain-specific skills)
predefined_skills = ["Python", "Machine Learning", "Data Analysis", "Power BI", "SQL"]

# Processing
if uploaded_file and job_description:
    resume_text = extract_text_from_pdf(uploaded_file)
    resume_text = preprocess_text(resume_text)
    doc = nlp(resume_text)
    
    # Extract entities
    entities = extract_entities(doc)
    
    # Match skills
    skills = match_skills(resume_text, predefined_skills)
    
    # Calculate similarity
    similarity_score = calculate_similarity(resume_text, job_description)
    
    # Display Results
    st.header("Resume Analysis")
    st.subheader("Extracted Information")
    st.write("**Name**:", entities["name"])
    st.write("**Contact**:", entities["contact"])
    st.write("**Skills**:", ', '.join(skills))
    st.write("**Education**:", ', '.join(entities["education"]))
    st.write("**Experience**:", ', '.join(entities["experience"]))
    
    st.subheader("Job Match Score")
    st.write(f"**Similarity with Job Description**: {similarity_score * 100:.2f}%")
    
    # Optional scoring system based on criteria
    scoring = similarity_score * 0.6 + len(skills) / len(predefined_skills) * 0.4
    st.write(f"**Overall Fit Score**: {scoring * 100:.2f}%")
    
    # Download option for analysis results
    analysis_df = pd.DataFrame([{
        "Name": entities["name"],
        "Contact": entities["contact"],
        "Skills": ', '.join(skills),
        "Education": ', '.join(entities["education"]),
        "Experience": ', '.join(entities["experience"]),
        "Job Match Score": f"{similarity_score * 100:.2f}%",
        "Overall Fit Score": f"{scoring * 100:.2f}%"
    }])
    
    csv = analysis_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="resume_analysis.csv">Download analysis as CSV</a>'
    st.markdown(href, unsafe_allow_html=True)
else:
    st.write("Please upload a resume and enter a job description to proceed.")
