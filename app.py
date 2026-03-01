import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("aml_lr_model.pkl")
scaler = joblib.load("aml_scaler.pkl")

# 👉 YAHAN ADD KARO 👇
st.write("Scaler expecting:", scaler.n_features_in_)

st.title("Financial Crime Detection")

inputs = []

for i in range(12):
    inputs.append(st.number_input(f"Feature {i+1}", value=0.0))

if st.button("Predict"):
    input_array = np.array(inputs).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠ Laundering Detected")
    else:
        st.success("✅ Normal Transaction")
