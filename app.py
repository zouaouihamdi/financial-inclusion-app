import streamlit as st
import pandas as pd
import joblib

model = joblib.load("financial_model.pkl")

st.title("Financial Inclusion Prediction")

st.subheader("Enter customer information")

country = st.selectbox("Country", ["Kenya", "Rwanda", "Tanzania", "Uganda"])
year = st.number_input("Year", 2016, 2025)
location_type = st.selectbox("Location Type", ["Urban", "Rural"])
cellphone_access = st.selectbox("Cellphone Access", ["Yes", "No"])
household_size = st.number_input("Household Size", 1, 20)
age = st.number_input("Age", 16, 100)
gender = st.selectbox("Gender", ["Male", "Female"])
relationship = st.selectbox("Relationship with Head", [
    "Head of Household",
    "Spouse",
    "Child",
    "Parent",
    "Other relative",
    "Other non-relatives"
])
marital_status = st.selectbox("Marital Status", [
    "Married/Living together",
    "Single/Never Married",
    "Widowed",
    "Divorced/Seperated"
])
education = st.selectbox("Education Level", [
    "No formal education",
    "Primary education",
    "Secondary education",
    "Vocational/Specialised training",
    "Tertiary education",
    "Other/Dont know/RTA"
])
job = st.selectbox("Job Type", [
    "Self employed",
    "Government Dependent",
    "Formally employed Private",
    "Informally employed",
    "Formally employed Government",
    "Farming and Fishing",
    "Remittance Dependent",
    "Other Income",
    "No Income",
    "Dont Know/Refuse to answer"
])

if st.button("Predict"):
    input_data = pd.DataFrame([{
        "country": country,
        "year": year,
        "location_type": location_type,
        "cellphone_access": cellphone_access,
        "household_size": household_size,
        "age_of_respondent": age,
        "gender_of_respondent": gender,
        "relationship_with_head": relationship,
        "marital_status": marital_status,
        "education_level": education,
        "job_type": job
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"This person is likely to have a bank account (Probability: {probability:.2f})")
    else:
        st.error(f"This person is NOT likely to have a bank account (Probability: {probability:.2f})")
