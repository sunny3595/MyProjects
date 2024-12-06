import streamlit as st
import docx2txt
import pdfplumber
import os
import re
from tika import parser
import spacy

# Load the SpaCy model
nlp = spacy.load("en_core_web_sm")

# Define skill sets for classification
SKILLS = [
    "Python", "Java", "SQL", "C++", "Machine Learning", "Data Science", 
    "Deep Learning", "HTML", "CSS", "JavaScript", "React", "Django", 
    "AWS", "Azure", "Docker", "Kubernetes", "DevOps", "Tableau", 
    "Power BI", "Excel"
]

def extract_text_from_file(file):
    file_extension = file.name.split(".")[-1].lower()
    if file_extension == "pdf":
        with pdfplumber.open(file) as pdf:
            text = "".join([page.extract_text() for page in pdf.pages])
    elif file_extension == "docx":
        text = docx2txt.process(file)
    elif file_extension == "txt":
        text = file.read().decode("utf-8")
    else:
        raw = parser.from_file(file)
        text = raw["content"]
    return text

def extract_skills(text):
    doc = nlp(text)
    extracted_skills = []
    for token in doc.ents:
        if token.text in SKILLS:
            extracted_skills.append(token.text)
    return list(set(extracted_skills))

# Streamlit App
st.title("Resume Classification")
st.markdown("### Upload a Resume to Extract Skills")

# Upload file section
uploaded_file = st.file_uploader("Drag and drop or browse a file (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
if uploaded_file:
    file_details = {
        "filename": uploaded_file.name,
        "filetype": uploaded_file.type,
        "filesize": uploaded_file.size
    }
    st.write("### File Details")
    st.json(file_details)

    # Extract text from the uploaded resume
    text = extract_text_from_file(uploaded_file)
    st.write("### Extracted Text (Preview)")
    st.text(text[:1000])  # Show first 1000 characters for preview

    # Process and display extracted skills
    if st.button("Extract Skills"):
        skills = extract_skills(text)
        if skills:
            st.success("### Extracted Skills:")
            st.write(skills)
        else:
            st.warning("No skills found in the uploaded resume.")
