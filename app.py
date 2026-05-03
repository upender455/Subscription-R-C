import pickle
import pandas as pd
import streamlit as st

# Load model & scaler
@st.cache_resource
def load_files():
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return scaler, model
    except Exception as e:
        st.error(f"File loading error: {e}")
        return None, None

scaler, model = load_files()

# UI
st.title("Customer Churn Prediction")

age = st.number_input("Age", 18, 100, 30)
country = st.selectbox("Country", ["USA", "Canada", "UK", "Germany", "France"])
subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
monthly_fee = st.number_input("Monthly Fee", value=9.99)
last_login_days = st.number_input("Days Since Last Login", value=30)
avg_usage = st.number_input("Avg Monthly Usage Hours", value=10.0)
support_tickets = st.number_input("Support Tickets (6m)", value=1)
auto_renew = st.selectbox("Auto Renew Enabled", [0, 1])
payment_method = st.selectbox("Payment Method", ["Credit Card", "PayPal", "Bank Transfer"])
cancelled = st.selectbox("Cancelled", [0, 1])

# Prediction button
if st.button("Predict Churn"):

    if scaler is None or model is None:
        st.error("Model or scaler not loaded properly.")
    else:
        # Create input dataframe AFTER user input
        data = {
            'age':[age],
            'country':[country],
            'subscription_type':[subscription_type],
            'monthly_fee':[monthly_fee],
            'last_login_days':[last_login_days],
            'avg_monthly_usage_hours':[avg_usage],
            'support_tickets_last_6m':[support_tickets],
            'auto_renew_enabled':[auto_renew],
            'payment_method':[payment_method],
            'cancelled':[cancelled]
        }

        df = pd.DataFrame(data)

        try:
            X_scaled = scaler.transform(df)
            pred = model.predict(X_scaled)

            result = "Yes" if pred[0] == 1 else "No"
            st.success(f"Predicted Churn: {result}")

        except Exception as e:
            st.error(f"Prediction error: {e}")