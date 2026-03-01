import streamlit as st
import joblib
import numpy as np

# ===============================
# Load Model & Scaler
# ===============================

model = joblib.load("aml_lr_model.pkl")
scaler = joblib.load("aml_scaler.pkl")

st.title("Financial Crime Detection System")

st.write("Enter Transaction Features")

# ===============================
# 12 Feature Inputs (Safe Default 0)
# ===============================

inputs = []

for i in range(12):
    inputs.append(
        st.number_input(f"Feature {i+1}", value=0.0)
    )

# ===============================
# Prediction Button
# ===============================

if st.button("Predict Fraud"):

    try:
        input_array = np.array(inputs).reshape(1, -1)

        # Scale Input
        input_scaled = scaler.transform(input_array)

        # Prediction
        prediction = model.predict(input_scaled)

        # Result Display
        if prediction[0] == 1:
            st.error("⚠ Laundering Detected")
        else:
            st.success("✅ Normal Transaction")

    except Exception as e:
        st.error("Prediction Error")
