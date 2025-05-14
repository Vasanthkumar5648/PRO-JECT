# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Load and preprocess data
data = pd.read_csv(r"D:\download\Telco_Customer_Churn (1).csv")
data.drop(columns=['customerID'], inplace=True)

# Convert categorical features
le = LabelEncoder()
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = le.fit_transform(data[column])

# Fill missing values
data.fillna(data.mean(), inplace=True)

# Split data
X = data.drop(columns=['Churn'])
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save artifacts
joblib.dump(model, 'churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')  # Save label encoder for categorical features
print("Model artifacts saved successfully")

# app.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import requests  # For CRM integration

# Page config
st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")

# Title
st.title("🚨 Customer Churn Prediction System")

# Sidebar for model management
with st.sidebar:
    st.header("Model Management")
    if st.button("🔄 Retrain Model"):
        with st.spinner("Training model..."):
            # Place your training code here (from train_model.py)
            # This would typically be in a separate file
            st.success("Model retrained successfully!")

    st.markdown("---")
    st.info("""
    **Instructions:**
    1. Fill customer details
    2. Click 'Predict Churn'
    3. High-risk customers will auto-alert CRM
    """)

# Initialize session state
if 'model' not in st.session_state:
    try:
        st.session_state.model = joblib.load('churn_model.pkl')
        st.session_state.scaler = joblib.load('scaler.pkl')
        st.session_state.encoder = joblib.load('label_encoder.pkl')
    except:
        st.warning("Model not found! Please train first.")
        st.session_state.model = None

# Prediction form
with st.form("prediction_form"):
    st.header("Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

    with col2:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0)

    submitted = st.form_submit_button("🔮 Predict Churn")

# Prediction logic
if submitted and st.session_state.model:
    # Prepare input data
    input_data = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
        # Add all other features needed by your model
    }])

    # Encode categorical features
    for col in input_data.select_dtypes(include=['object']).columns:
        input_data[col] = st.session_state.encoder.transform(input_data[col])

    # Scale features
    scaled_data = st.session_state.scaler.transform(input_data)

    # Predict
    prediction = st.session_state.model.predict(scaled_data)[0]
    probability = st.session_state.model.predict_proba(scaled_data)[0][1]

    # Display results
    st.subheader("Prediction Results")

    if prediction == 1:
        st.error(f"⛔ Churn Risk: {probability:.1%} (High Risk)")
    else:
        st.success(f"✅ Churn Risk: {probability:.1%} (Low Risk)")

    # Visualize probability
    st.progress(probability)

    # CRM Integration (example with webhook)
    if probability > 0.7:  # High risk threshold
        with st.spinner("Alerting CRM..."):
            crm_response = notify_crm(input_data, probability)

        if "success" in crm_response.lower():
            st.toast("📢 CRM Alert Sent!", icon="✅")
        else:
            st.warning("CRM notification failed")


# CRM Notification Function
def notify_crm(customer_data, probability):
    """Example integration with CRM system"""
    try:
        # Replace with your actual CRM integration
        crm_webhook = "https://your-crm-api.com/alerts"

        payload = {
            "customer": customer_data.to_dict(orient='records')[0],
            "churn_risk": probability,
            "alert_timestamp": pd.Timestamp.now().isoformat()
        }

        response = requests.post(crm_webhook, json=payload, timeout=5)
        return response.text

    except Exception as e:
        return f"Error: {str(e)}"


# Model performance section
st.markdown("---")
st.header("📊 Model Performance")

if st.session_state.model:
    # Placeholder for model metrics - replace with actual metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", "87%", "2%")
    col2.metric("Precision", "83%", "-1%")
    col3.metric("Recall", "76%", "3%")

    # Feature importance visualization
    if hasattr(st.session_state.model, 'feature_importances_'):
        st.subheader("Feature Importance")
        features = pd.DataFrame({
            'Feature': X_train.columns,  # Replace with your feature names
            'Importance': st.session_state.model.feature_importances_
        }).sort_values('Importance', ascending=False)

        st.bar_chart(features.set_index('Feature'))
else:
    st.warning("No model loaded - train a model first")

# How to Run:
# 1. Save as app.py
# 2. Run with: streamlit run app.py