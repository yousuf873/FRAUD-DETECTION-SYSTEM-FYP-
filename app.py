import streamlit as st
import joblib
import numpy as np

model = joblib.load("aml_lr_model.pkl")
scaler = joblib.load("aml_scaler.pkl")

st.title("Financial Crime Detection System")

st.write("Enter Transaction Features")

# ⭐ Must match training feature size = 5194

inputs = []

for i in range(5194):
    inputs.append(st.number_input(f"F{i+1}", value=0.0))

if st.button("Predict Fraud"):

    input_array = np.array(inputs).reshape(1, -1)

    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)

    if prediction[0] > 0.5:
        st.error("⚠ Fraud Detected")
    else:
        st.success("✅ Normal Transaction")
