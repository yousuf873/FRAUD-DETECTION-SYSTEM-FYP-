import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model         = joblib.load(os.path.join(BASE_DIR, "aml_lr_model.pkl"))
scaler        = joblib.load(os.path.join(BASE_DIR, "aml_scaler.pkl"))
label_encoders = joblib.load(os.path.join(BASE_DIR, "aml_label_encoders.pkl"))
features      = joblib.load(os.path.join(BASE_DIR, "aml_features.pkl"))

st.set_page_config(page_title="Financial Crime Detection", page_icon="🔍", layout="wide")
st.title("🔍 Financial Crime Detection System")
st.caption("AML — Anti Money Laundering Transaction Analyzer")

with st.sidebar:
    st.header("ℹ️ About")
    st.info("Enter transaction details to check if it is suspicious or clean.")
    st.metric("Model", "Logistic Regression")
    st.metric("Features Used", "8")

st.subheader("Transaction Details")
col1, col2 = st.columns(2)

with col1:
    amount           = st.number_input("💰 Transaction Amount", min_value=0.0, value=1000.0, step=100.0)
    sender_account   = st.number_input("👤 Sender Account Number", min_value=0, value=10000000)
    receiver_account = st.number_input("🏦 Receiver Account Number", min_value=0, value=20000000)
    payment_type     = st.selectbox("💳 Payment Type", label_encoders['Payment_type'].classes_)

with col2:
    sender_location   = st.selectbox("📍 Sender Bank Location", label_encoders['Sender_bank_location'].classes_)
    receiver_location = st.selectbox("📍 Receiver Bank Location", label_encoders['Receiver_bank_location'].classes_)
    payment_currency  = st.selectbox("💱 Payment Currency", label_encoders['Payment_currency'].classes_)
    received_currency = st.selectbox("💱 Received Currency", label_encoders['Received_currency'].classes_)

if st.button("🔎 Analyze Transaction", use_container_width=True):

    encoded = {
        'Amount':                amount,
        'Sender_account':        sender_account,
        'Receiver_account':      receiver_account,
        'Payment_type':          label_encoders['Payment_type'].transform([payment_type])[0],
        'Sender_bank_location':  label_encoders['Sender_bank_location'].transform([sender_location])[0],
        'Receiver_bank_location':label_encoders['Receiver_bank_location'].transform([receiver_location])[0],
        'Payment_currency':      label_encoders['Payment_currency'].transform([payment_currency])[0],
        'Received_currency':     label_encoders['Received_currency'].transform([received_currency])[0],
    }

    input_df     = pd.DataFrame([encoded])[features]
    input_scaled = scaler.transform(input_df)

    prediction  = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Risk Score",  f"{probability:.1%}")
    c2.metric("Prediction",  "SUSPICIOUS 🚨" if prediction == 1 else "CLEAN ✅")
    c3.metric("Confidence",  f"{max(probability, 1-probability):.1%}")

    if prediction == 1:
        st.error(f"🚨 Suspicious Transaction Detected! Risk Score: {probability:.1%}")
    else:
        st.success(f"✅ Transaction appears clean. Risk Score: {probability:.1%}")

    st.progress(float(probability))

    st.subheader("Transaction Summary")
    st.table(pd.DataFrame({
        "Field": ["Amount","Sender Account","Receiver Account","Payment Type",
                  "Sender Location","Receiver Location","Payment Currency","Received Currency"],
        "Value": [amount, sender_account, receiver_account, payment_type,
                  sender_location, receiver_location, payment_currency, received_currency]
    }))

    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({
        "Amount": amount, "From": sender_location, "To": receiver_location,
        "Currency": payment_currency,
        "Risk Score": f"{probability:.1%}",
        "Result": "SUSPICIOUS" if prediction == 1 else "CLEAN"
    })

st.divider()

if "history" in st.session_state and st.session_state.history:
    st.subheader("📋 Transaction History (This Session)")
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df, use_container_width=True)
    csv = hist_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download History as CSV", csv, "transaction_history.csv", "text/csv")
