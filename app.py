import streamlit as st
import joblib
import numpy as np

model = joblib.load("aml_lr_model.pkl")
scaler = joblib.load("aml_scaler.pkl")
pca = joblib.load("aml_pca.pkl")

st.title("Financial Crime Detection System")

inputs = []

for i in range(5194):
    inputs.append(st.number_input(f"Feature {i+1}", value=0.0))

if st.button("Predict"):

    try:
        x = np.array(inputs).reshape(1,-1)

        x_pca = pca.transform(x)
        x_scaled = scaler.transform(x_pca)

        prediction = model.predict(x_scaled)

        if prediction[0]==1:
            st.error("Laundering Detected ⚠")
        else:
            st.success("Normal Transaction ✅")

    except:
        st.error("Prediction Error")
