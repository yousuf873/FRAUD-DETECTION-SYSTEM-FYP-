import streamlit as st
import joblib
import numpy as np

model  = joblib.load("aml_lr_model.pkl")
scaler = joblib.load("aml_scaler.pkl")
pca    = joblib.load("aml_pca.pkl")

st.set_page_config(page_title="AML Detection", page_icon="🏦")
st.title("🏦 Financial Crime Detection System")
st.write("Enter transaction details below:")

col1, col2 = st.columns(2)

with col1:
    amount           = st.number_input("💰 Transaction Amount", min_value=0.0, value=1000.0)
    sender_account   = st.number_input("👤 Sender Account", min_value=0, value=1234567890)
    receiver_account = st.number_input("👤 Receiver Account", min_value=0, value=9876543210)
    payment_type     = st.selectbox("💳 Payment Type",
                         ["Cash Deposit", "Cross-border", "Cheque", "ACH", "Wire Transfer"])

with col2:
    sender_location   = st.selectbox("📍 Sender Bank Location",
                          ["UK", "UAE", "US", "EU", "Mexico", "Other"])
    receiver_location = st.selectbox("📍 Receiver Bank Location",
                          ["UK", "UAE", "US", "EU", "Mexico", "Other"])
    payment_currency  = st.selectbox("💱 Payment Currency",
                          ["UK pounds", "Dirham", "US Dollar", "Euro", "Other"])
    received_currency = st.selectbox("💱 Received Currency",
                          ["UK pounds", "Dirham", "US Dollar", "Euro", "Other"])

payment_map  = {"Cash Deposit": 0, "Cross-border": 1, "Cheque": 2, "ACH": 3, "Wire Transfer": 4}
location_map = {"UK": 0, "UAE": 1, "US": 2, "EU": 3, "Mexico": 4, "Other": 5}
currency_map = {"UK pounds": 0, "Dirham": 1, "US Dollar": 2, "Euro": 3, "Other": 4}

if st.button("🔍 Analyse Transaction"):
    raw = np.array([[
        amount,
        sender_account,
        receiver_account,
        payment_map[payment_type],
        location_map[sender_location],
        location_map[receiver_location],
        currency_map[payment_currency],
        currency_map[received_currency]
    ]])

    # Correct order: raw(8) → PCA(3) → scale → predict
    x_pca    = pca.transform(raw)
    x_scaled = scaler.transform(x_pca)

    prediction  = model.predict(x_scaled)
    probability = model.predict_proba(x_scaled)[0][1]

    st.divider()
    if prediction[0] == 1:
        st.error(f"🚨 Laundering Detected ⚠️ — Confidence: {probability:.1%}")
    else:
        st.success(f"✅ Normal Transaction — Fraud Probability: {probability:.1%}")
