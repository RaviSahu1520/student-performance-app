# =====================================================
# 🎓 Student Performance Prediction - Streamlit App (Linear Regression)
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# ---------------------------
# Page Setup
# ---------------------------
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("🎓 Student Performance Prediction App")
st.write("Train and use a Linear Regression model to predict student performance based on various academic and behavioral factors.")

# ---------------------------
# Load Dataset
# ---------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Student_Performance.csv")

try:
    data = load_data()
    st.subheader("📊 Dataset Overview")
    st.dataframe(data.head())
except FileNotFoundError:
    st.error("⚠️ 'Student_Performance.csv' file not found in this directory.")
    st.stop()

# ---------------------------
# Model Training Section
# ---------------------------
st.markdown("### 🔧 Model Training")

target_col = st.selectbox("🎯 Select Target Column (Performance / Score)", data.columns)
feature_cols = [c for c in data.columns if c != target_col]

X = data[feature_cols].copy()
y = data[target_col].copy()

# Encode categorical columns
label_encoders = {}
for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

# Encode target if categorical
if y.dtype == "object":
    y_le = LabelEncoder()
    y = y_le.fit_transform(y)
else:
    y_le = None

# Train Model Button
if st.button("🚀 Train Model"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    st.success(f"✅ Model trained successfully using Linear Regression! R² Score: **{score:.2f}**")

    # Save model
    with open("student_model.pkl", "wb") as f:
        pickle.dump((model, label_encoders, y_le, feature_cols, target_col), f)

    st.download_button(
        label="📥 Download Trained Model (.pkl)",
        data=open("student_model.pkl", "rb"),
        file_name="student_model.pkl",
        mime="application/octet-stream",
    )

# ---------------------------
# Load Trained Model Section
# ---------------------------
st.markdown("### 📂 Load Existing Model")

model_path = "student_model.pkl"

if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model, label_encoders, y_le, feature_cols, target_col = pickle.load(f)
    st.success("✅ Existing model loaded successfully!")
else:
    st.info("ℹ️ No saved model found yet. Train the model first to create one.")

# ---------------------------
# Prediction Section
# ---------------------------
if "model" in locals():
    st.markdown("### 🧮 Make a Prediction")

    input_data = {}
    for col in feature_cols:
        if data[col].dtype == "object":
            val = st.selectbox(f"{col}", data[col].unique())
        else:
            val = st.number_input(
                f"{col}",
                float(data[col].min()),
                float(data[col].max()),
                float(data[col].mean()),
            )
        input_data[col] = val

    if st.button("🔍 Predict Performance"):
        df = pd.DataFrame([input_data])

        # Apply same encoding as training
        for col, le in label_encoders.items():
            df[col] = le.transform(df[col])

        prediction = model.predict(df)[0]

        # Decode prediction if label encoded
        if y_le:
            prediction = y_le.inverse_transform([int(round(prediction))])[0]

        st.subheader(f"🎯 Predicted {target_col}: **{prediction}**")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
