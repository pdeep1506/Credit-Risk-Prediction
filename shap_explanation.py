import pandas as pd
import shap



def format_feature_name(feature):

    feature = feature.replace("num__", "")
    feature = feature.replace("cat__", "")

    mapping = {

        "person_age":
            "Age",

        "person_income":
            "Annual Income",

        "person_emp_length":
            "Employment Length",

        "loan_amnt":
            "Loan Amount",

        "loan_int_rate":
            "Interest Rate",

        "loan_percent_income":
            "Loan % of Income",

        "cb_person_cred_hist_length":
            "Credit History Length",

        "person_home_ownership_RENT":
            "Home Ownership: Rent",

        "person_home_ownership_MORTGAGE":
            "Home Ownership: Mortgage",

        "person_home_ownership_OWN":
            "Home Ownership: Own",

        "person_home_ownership_OTHER":
            "Home Ownership: Other",

        "loan_intent_EDUCATION":
            "Loan Purpose: Education",

        "loan_intent_MEDICAL":
            "Loan Purpose: Medical",

        "loan_intent_VENTURE":
            "Loan Purpose: Business / Venture",

        "loan_intent_PERSONAL":
            "Loan Purpose: Personal",

        "loan_intent_DEBTCONSOLIDATION":
            "Loan Purpose: Debt Consolidation",

        "loan_intent_HOMEIMPROVEMENT":
            "Loan Purpose: Home Improvement",

        "cb_person_default_on_file_N":
            "Previous Default: No",

        "cb_person_default_on_file_Y":
            "Previous Default: Yes"
    }

    return mapping.get(feature, feature)



def get_shap_explanation(model, input_data, prediction):

   
    # Get trained preprocessing pipeline
    

    preprocessor = model.named_steps["preprocessor"]


    # Get trained XGBoost model
   

    xgb_model = model.named_steps["model"]

    
    # Transform input using trained preprocessor
    

    input_transformed = preprocessor.transform(
        input_data
    )

   
    # Get feature names after OneHotEncoding
   

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

   
    # Calculate SHAP values
 

    explainer = shap.TreeExplainer(
        xgb_model
    )

    shap_values = explainer(
        input_transformed
    )

    shap_array = shap_values.values[0]

    
    # Create SHAP DataFrame
   

    shap_df = pd.DataFrame({

        "Feature": feature_names,

        "SHAP Value": shap_array

    })


    
    # IMPORTANT
    # Only keep categorical feature that was actually selected
  

    categorical_groups = {

        "person_home_ownership": (
            "person_home_ownership"
        ),

        "loan_intent": (
            "loan_intent"
        ),

        "cb_person_default_on_file": (
            "cb_person_default_on_file"
        )
    }


    rows_to_keep = []


    for _, row in shap_df.iterrows():

        feature = row["Feature"]

        # Remove preprocessing prefixes
        clean_feature = feature.replace(
            "cat__",
            ""
        ).replace(
            "num__",
            ""
        )


        
        # Home Ownership
        

        if clean_feature.startswith(
            "person_home_ownership_"
        ):

            selected_value = (
                input_data[
                    "person_home_ownership"
                ].iloc[0]
            )

            expected_feature = (
                f"person_home_ownership_{selected_value}"
            )

            if clean_feature == expected_feature:

                rows_to_keep.append(row)


        
        # Loan Intent
        

        elif clean_feature.startswith(
            "loan_intent_"
        ):

            selected_value = (
                input_data[
                    "loan_intent"
                ].iloc[0]
            )

            expected_feature = (
                f"loan_intent_{selected_value}"
            )

            if clean_feature == expected_feature:

                rows_to_keep.append(row)


       
        # Previous Default
 

        elif clean_feature.startswith(
            "cb_person_default_on_file_"
        ):

            selected_value = (
                input_data[
                    "cb_person_default_on_file"
                ].iloc[0]
            )

            expected_feature = (
                f"cb_person_default_on_file_{selected_value}"
            )

            if clean_feature == expected_feature:

                rows_to_keep.append(row)


      
        # Numerical features
 

        else:

            rows_to_keep.append(row)


   
    # Create filtered DataFrame
    

    filtered_shap_df = pd.DataFrame(
        rows_to_keep
    )



    # Convert feature names
    

    filtered_shap_df["Feature"] = (
        filtered_shap_df["Feature"]
        .apply(format_feature_name)
    )


    
    # HIGH RISK
   

    if prediction == 1:

        top_5 = (

            filtered_shap_df[
                filtered_shap_df["SHAP Value"] > 0
            ]

            .sort_values(
                by="SHAP Value",
                ascending=False
            )

            .head(5)
        )

        title = (
            "⚠️ Top 5 Factors Increasing Credit Risk"
        )

        impact = (
            "increased the predicted risk"
        )


   
    # LOW RISK
   

    else:

        top_5 = (

            filtered_shap_df[
                filtered_shap_df["SHAP Value"] < 0
            ]

            .sort_values(
                by="SHAP Value",
                ascending=True
            )

            .head(5)
        )

        title = (
            "✅ Top 5 Factors Supporting Low Credit Risk"
        )

        impact = (
            "reduced the predicted risk"
        )


    return top_5, title, impact