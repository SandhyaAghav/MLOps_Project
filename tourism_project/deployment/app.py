import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="Sandhya-2025/product-purchase-model", filename="best_product_purchase_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Tourism product purchase App")
st.write("The Product Purchase Prediction App is an internal tool that predicts whether customer will purchase the newly introduced Wellness Tourism Package before contacting them.")
st.write("Kindly enter the customer details to check whether customer will purchase the newly introduced Wellness Tourism Package")

# Collect user input
TypeofContact = st.selectbox("Select the method by which the customer was contacted", ["Company Invited", "Self Inquiry"])
Age = st.number_input("Enter the age (customer's age in years)", min_value=18, max_value=100, value=30)
CityTier=st.number_input("Enter the city category based on development, population, and living standards (Tier 1 > Tier 2 > Tier 3)",min_value=1,max_value=3, value=2)
Occupation = st.selectbox("Select Customer's occupation", ["Salaried", "Freelancer"])
Gender = st.selectbox("Select gender of the customer", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Enter the total number of people accompanying the customer on the trip.", min_value=0, value=1)
MonthlyIncome = st.number_input("Enter gross monthly income of the customer.", min_value=1000.0, value=50000.0)
PreferredPropertyStar = st.number_input("Enter the preferred hotel rating by the customer.)", min_value=3,max_value=5, value=4)
MaritalStatus = st.selectbox("Select marital status of the customer.", ["Single", "Married","Divorced"])
NumberOfTrips = st.number_input("Enter the average number of trips the customer takes annually.", min_value=1,max_value=24, value=5)
Passport = st.selectbox("Whether the customer holds a valid passport?", ["Yes", "No"])
OwnCar = st.selectbox("Whether the customer owns a car?", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Enter the number of children below age 5 accompanying the customer.", min_value=0,max_value=4, value=0)
Designation = st.selectbox("Select the customer's designation in their current organization.", ["Executive", "Manager","Senior Manager","AVP","VP"])
PitchSatisfactionScore = st.number_input("Enter the score indicating the customer's satisfaction with the sales pitch.)", min_value=1,max_value=5, value=3)
ProductPitched = st.selectbox("Select the type of product pitched to the customer.", ["Basic", "Deluxe","Standard","Super Deluxe","King"])
NumberOfFollowups =st.number_input("Enter the total number of follow-ups by the salesperson after the sales pitch.", min_value=1,max_value=10, value=2)
DurationOfPitch = st.number_input("Enter the duration of the sales pitch delivered to the customer.", min_value=5,max_value=127, value=10)


# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'Gender': Gender,
    'Designation': Designation,
    'MaritalStatus': MaritalStatus,
    'ProductPitched': ProductPitched,
    'Age':Age,
    'CityTier':CityTier,
    'NumberOfPersonVisiting' : NumberOfPersonVisiting,
    'MonthlyIncome' : MonthlyIncome,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Passport': 1 if Passport == "Yes" else 0,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfFollowups': NumberOfFollowups,
    'DurationOfPitch': DurationOfPitch,
    'PitchSatisfactionScore': PitchSatisfactionScore
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "Customer will purchase the product" if prediction == 1 else "Customer will not purchase the product"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
