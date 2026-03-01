
import streamlit as st
import joblib
import numpy as np

# Load Model
model = joblib.load("aml_lr_model.pkl")
scaler = joblib.load("aml_scaler.pkl")

st.title("Financial Crime Detection System")

st.write("Enter Transaction Features")

inputs = []

for i in range(12):
    val = st.number_input(f"Feature {i+1}")
    inputs.append(val)

if st.button("Predict Fraud"):

    input_array = np.array([inputs])

    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)

    if prediction[0] > 0.5:
        st.error(" Fraud Transaction Detected")
    else:
        st.success(" Normal Transaction")
