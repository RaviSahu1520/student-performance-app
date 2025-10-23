# student_performance_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# =============================
# LOAD DATA AND TRAIN MODEL
# =============================

# Load dataset
data = pd.read_csv("Student_Performance.csv")

# Encode categorical column
encoder = LabelEncoder()
data["Extracurricular Activities"] = encoder.fit_transform(data["Extracurricular Activities"])

# Split data
X = data.drop(columns="Performance Index")
y = data["Performance Index"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# =============================
# STREAMLIT UI
# =============================

st.set_page_config(page_title="Student Performance Predictor", page_icon="📊", layout="centered")

st.title("🎓 Student Performance Prediction App")
st.write("Enter student details below to predict their **Performance Index**.")

# Input fields
hours_studied = st.number_input("📘 Hours Studied", min_value=0, max_value=24, value=6)
previous_scores = st.number_input("📊 Previous Scores", min_value=0, max_value=100, value=70)
extracurricular = st.selectbox("🏅 Extracurricular Activities", options=["Yes", "No"])
sleep_hours = st.number_input("😴 Sleep Hours", min_value=0, max_value=24, value=8)
papers_practiced = st.number_input("📄 Sample Question Papers Practiced", min_value=0, max_value=20, value=3)

# Prepare input
extracurricular_encoded = encoder.transform([extracurricular])[0]
input_data = np.array([[hours_studied, previous_scores, extracurricular_encoded, sleep_hours, papers_practiced]])

# Predict button
if st.button("🔍 Predict Performance"):
    prediction = np.round(model.predict(input_data)[0], 2)

    st.success(f"🎯 Predicted Performance Index: **{prediction}**")

    # Optionally display model evaluation
    r2 = r2_score(y_test, model.predict(X_test))
    mae = mean_absolute_error(y_test, model.predict(X_test))
    st.write("---")
    st.write("📈 **Model Performance Metrics:**")
    st.metric("R² Score", f"{r2:.3f}")
    st.metric("Mean Absolute Error", f"{mae:.3f}")

# Add small footer
st.write("---")
st.caption("Developed with ❤️ using Streamlit | Powered by Linear Regression")
