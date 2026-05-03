import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import shap
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Explainable AI: Loan Approval",
    page_icon="💰",
    layout="wide"
)

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Cache data/model loading
@st.cache_resource
def load_model_assets():
    path = os.path.join('models', 'loan_model_assets.pkl')
    if os.path.exists(path):
        return joblib.load(path)
    return None

assets = load_model_assets()

def main():
    st.title("🏦 Explainable AI for Loan Approval")
    st.markdown("---")

    if assets is None:
        st.error("Model assets not found! Please run `python scripts/train_model.py` first.")
        return

    model = assets['model']
    scaler = assets['scaler']
    label_encoders = assets['label_encoders']
    feature_names = assets['feature_names']

    # Layout: Sidebar for inputs, Main for results
    with st.sidebar:
        st.header("📋 Applicant Details")
        
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        
        income = st.number_input("Applicant Income (₹)", min_value=0, value=5000, step=100)
        co_income = st.number_input("Coapplicant Income (₹)", min_value=0, value=0, step=100)
        loan_amt = st.number_input("Loan Amount (₹k)", min_value=0, value=150, step=10)
        term = st.selectbox("Loan Term (Months)", [120, 180, 240, 360, 480], index=3)
        credit = st.selectbox("Credit History", ["Clear", "Not Clear"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    # Prepare input data
    input_dict = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': income,
        'CoapplicantIncome': co_income,
        'LoanAmount': loan_amt,
        'Loan_Amount_Term': term,
        'Credit_History': 1.0 if credit == "Clear" else 0.0,
        'Property_Area': property_area
    }
    
    # Process input for model
    proc_dict = input_dict.copy()
    for col, le in label_encoders.items():
        if col in proc_dict and col != 'Loan_Status':
            try:
                # Handle cases where value might be new (though selectbox prevents this)
                proc_dict[col] = le.transform([proc_dict[col]])[0]
            except Exception:
                # Fallback to first class if error (shouldn't happen with selectboxes)
                proc_dict[col] = 0

    # Convert to DataFrame and Scale
    input_df = pd.DataFrame([proc_dict])[feature_names]
    input_scaled = scaler.transform(input_df)

    # Prediction
    if st.button("Analyze Loan Application"):
        col1, col2 = st.columns([1, 2])
        
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        with col1:
            st.subheader("Decision")
            if prediction == 1:
                st.success(f"✅ Approved\n\nProbability: {probability:.1%}")
            else:
                st.error(f"❌ Denied\n\nProbability: {probability:.1%}")
                
            st.info("""
            **What is SHAP?**
            SHAP (SHapley Additive exPlanations) is a method to explain individual predictions. 
            It shows how much each feature contributed to the final outcome compared to the average prediction.
            """)

        with col2:
            st.subheader("🔍 Explainable AI (SHAP)")
            
            # Setup SHAPExplainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_scaled)
            
            # Random Forest SHAP output can be a list (for classification) or array
            # We want the values for class 1 (Approved)
            if isinstance(shap_values, list):
                sv = shap_values[1]
            else:
                sv = shap_values[..., 1] # SHAP 0.45+ style

            # Create waterfall plot
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots._waterfall.waterfall_legacy(
                explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
                sv[0],
                feature_names=feature_names,
                max_display=10,
                show=False
            )
            plt.title("Contribution of Features to Prediction")
            st.pyplot(fig)
            
            st.write("---")
            st.markdown("**Interpreting the graph:**")
            st.markdown("- **Blue (left)**: Features that decreased the approval chance.")
            st.markdown("- **Red (right)**: Features that increased the approval chance.")

    # Global explanation section
    st.markdown("---")
    with st.expander("🌍 Global Model Insights"):
        st.subheader("Overall Feature Importance")
        # Global importance is fixed, can be pre-calculated or done on the fly with a sample
        # Here we use the model's internal importance for speed
        importances = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=True)
        
        fig2, ax2 = plt.subplots()
        ax2.barh(importances['Feature'], importances['Importance'], color='skyblue')
        ax2.set_title("Which features matter most overall?")
        st.pyplot(fig2)

if __name__ == "__main__":
    main()
