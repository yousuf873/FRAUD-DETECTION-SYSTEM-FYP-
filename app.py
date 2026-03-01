
import streamlit as st
import joblib
import numpy as np

model = joblib.load("aml_lr_model.pkl")
scaler = joblib.load("aml_scaler.pkl")

st.title("Financial Crime Detection System")

amount = st.number_input("Transaction Amount")
old_balance = st.number_input("Old Balance")
new_balance = st.number_input("New Balance")

if st.button("Predict Fraud"):

    input_data = np.array([[amount, old_balance, new_balance]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    if prediction[0] > 0.5:
        st.error("⚠ Fraud Detected")
    else:
        st.success("✅ Normal Transaction")
