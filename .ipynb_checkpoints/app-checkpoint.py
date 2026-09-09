from shap_explanation import get_shap_explanation
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/credit_risk_xgboost.pkl")

# Page configration
st.set_page_config(page_title = "Credit Risk Prediction",page_icon="💳",
    layout="centered")

# Application Title
st.title("💳 Credit Risk Prediction")

st.write(
     "Enter the applicant's information to predict credit risk."
)

# Input form
person_age = st.number_input("Age", min_value=18, max_value=100, value=25,step=1)

person_income = st.number_input("Annual Income", min_value=0, value=10000,step=10000)
home_ownership_display = st.selectbox(
    "Home Ownership",
    [
        "Rent",
        "Mortgage",
        "Own",
        "Other"
    ]
)

home_ownership_mapping = {
    "Rent": "RENT",
    "Mortgage": "MORTGAGE",
    "Own": "OWN",
    "Other": "OTHER"
}

person_home_ownership = home_ownership_mapping[
    home_ownership_display
]

person_emp_length = st.number_input("Employment Length (years)", min_value=0.0, max_value=70.0, value=2.0
                                    , step=0.5)



loan_intent_display = st.selectbox(
    "Loan Intent",
    [
        "Education",
        "Medical",
        "Business / Venture",
        "Personal",
        "Debt Consolidation",
        "Home Improvement"
    ]
)

# Convert UI value to the value used during model training
loan_intent_mapping = {
    "Education": "EDUCATION",
    "Medical": "MEDICAL",
    "Business / Venture": "VENTURE",
    "Personal": "PERSONAL",
    "Debt Consolidation": "DEBTCONSOLIDATION",
    "Home Improvement": "HOMEIMPROVEMENT"
}

loan_intent = loan_intent_mapping[loan_intent_display]

loan_amnt = st.number_input(
    "Loan Amount",
    min_value=0,
    value= 100, step=1000
)
# Calculate Loan % of Income
if person_income is not None and person_income > 0:
    loan_percent_income = loan_amnt / person_income

    st.info(
        f"Loan % of Income = "
        f"${loan_amnt:,.0f} ÷ ${person_income:,.0f} "
        f"= {loan_percent_income:.2%}"
    )

    # Show warning if loan is greater than annual income
    if loan_percent_income > 1:
        st.warning(
            f"Loan amount is {loan_percent_income:.2%} "
            "of annual income."
        )


loan_int_rate = st.number_input(
    "Interest Rate (%)",
    min_value=0.0,
    max_value=100.0,
    value=10.0, step=0.1
)

# Loadn to Income % calculation
col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Loan Amount",
        f"${loan_amnt:,.0f}"
    )

with col2:
    st.metric(
        "Annual Income",
        f"${person_income:,.0f}"
    )

# Calculate loan percentage of income
if person_income > 0:
    loan_percent_income = loan_amnt / person_income

    st.info(
        f"Loan % of Income = "
        f"${loan_amnt:,.0f} ÷ ${person_income:,.0f} "
        f"= {loan_percent_income:.2%}"
    )
else:
    loan_percent_income = None
    st.error("Annual Income must be greater than 0.")

cb_person_default_on_file = st.selectbox(
    "Previous Default",
    ["N", "Y"]
)


cb_person_cred_hist_length = st.number_input(
    "Credit History Length (years)",
    min_value=0,
    max_value=100,
    value=4, step=1
)

if st.button("Predict Credit Risk"):

    # Input validation
    if person_income <= 0:
        st.error("Annual Income must be greater than 0.")



    elif loan_amnt <= 0:
        st.error("Loan Amount must be greater than 0.")

    elif loan_int_rate <= 0:
        st.error("Interest Rate must be greater than 0.")

    elif person_age <= 0:
        st.error("Age must be greater than 0.")

    elif person_emp_length < 0:
        st.error("Employment Length cannot be negative.")

    elif cb_person_cred_hist_length < 0:
        st.error("Credit History Length cannot be negative.")
    else:
        input_data = pd.DataFrame({
        "person_age": [person_age],
        "person_income": [person_income],
        "person_home_ownership": [person_home_ownership],
        "person_emp_length": [person_emp_length],
        "loan_intent": [loan_intent],
       
        "loan_amnt": [loan_amnt],
        "loan_int_rate": [loan_int_rate],
        "loan_percent_income": [loan_percent_income],
        "cb_person_default_on_file": [
            cb_person_default_on_file
        ],
        "cb_person_cred_hist_length": [
            cb_person_cred_hist_length
        ]
    })

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(
        input_data
    )[0][1]

    st.subheader("Prediction Result")

    st.metric(
        "Default Probability",
        f"{probability:.2%}"
    )

    if prediction == 1:
        st.error("⚠️ High Credit Risk")
    else:
        st.success("✅ Low Credit Risk")

    # Display probability
    st.metric(
        "Default Probability",
        f"{probability:.2%}"
    )


        # SHAP explanation
    top_5, title, impact = get_shap_explanation(
        model,
        input_data,
        prediction
    )


    st.divider()

    st.subheader(title)

    for _, row in top_5.iterrows():

        st.write(
            f"**{row['Feature']}** "
            f"{impact}."
    )


# ============================================================
# Dataset & Model Information
# ============================================================

st.divider()

st.subheader("📊 About the Training Data")

st.write(
    """
    This Credit Risk Prediction model was trained using a dataset
    containing information about loan applicants and their loan
    characteristics.
    """
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", "32,581")

with col2:
    st.metric("Features Used", "11")

with col3:
    st.metric("Target Variable", "Loan Status")


st.markdown("### 🧾 Features Used for Training")

features = [
    "Person Age",
    "Person Income",
    "Home Ownership",
    "Employment Length",
    "Loan Intent",
    "Loan Amount",
    "Loan Interest Rate",
    "Loan Percent of Income",
    "Previous Default on File",
    "Credit History Length"
]

col1, col2, col3 = st.columns(3)

for i, feature in enumerate(features):
    if i % 3 == 0:
        col1.markdown(f"• {feature}")
    elif i% 2 == 0:
        col2.markdown(f"• {feature}")
    else:
        col3.markdown(f"• {feature}")



st.markdown("### 🤖 Machine Learning Model")

st.write(
    """
    The prediction model is an XGBoost classifier. The training
    pipeline includes data preprocessing for numerical and
    categorical features followed by the trained XGBoost model.
    """
)



st.caption(
    "Model trained using the Credit Risk dataset with preprocessing, "
    "XGBoost classification, hyperparameter tuning, and SHAP-based "
    "model explanations."
)


