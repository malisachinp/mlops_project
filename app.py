import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "models/best_model.joblib"
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Wellness Tourism Predictor", page_icon="✈️")
st.title("✈️ Wellness Tourism Package Purchase Predictor")
st.write("Predict whether a customer is likely to purchase the new Wellness Tourism Package.")

with st.form("prediction_form"):
    age = st.number_input("Age", 18, 100, 35)
    type_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    duration = st.number_input("Duration of Pitch (minutes)", 1.0, 60.0, 10.0)
    occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    persons = st.number_input("Number of Persons Visiting", 1, 20, 2)
    followups = st.number_input("Number of Followups", 0, 10, 3)
    product = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
    stars = st.selectbox("Preferred Property Star", [3, 4, 5])
    marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    trips = st.number_input("Number of Trips", 0.0, 30.0, 3.0)
    passport = st.selectbox("Passport", [0, 1], format_func=lambda x: "Yes" if x else "No")
    pitch_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
    own_car = st.selectbox("Own Car", [0, 1], format_func=lambda x: "Yes" if x else "No")
    children = st.number_input("Number of Children Visiting", 0.0, 10.0, 1.0)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    income = st.number_input("Monthly Income", 1000.0, 100000.0, 20000.0)
    submitted = st.form_submit_button("Predict Purchase")

if submitted:
    input_df = pd.DataFrame([{
        "Age": age, "TypeofContact": type_contact, "CityTier": city_tier,
        "DurationOfPitch": duration, "Occupation": occupation, "Gender": gender,
        "NumberOfPersonVisiting": persons, "NumberOfFollowups": followups,
        "ProductPitched": product, "PreferredPropertyStar": stars,
        "MaritalStatus": marital, "NumberOfTrips": trips, "Passport": passport,
        "PitchSatisfactionScore": pitch_score, "OwnCar": own_car,
        "NumberOfChildrenVisiting": children, "Designation": designation,
        "MonthlyIncome": income
    }])
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0, 1])
    st.subheader("Prediction")
    if prediction == 1:
        st.success(f"Likely to purchase — probability: {probability:.1%}")
    else:
        st.info(f"Unlikely to purchase — probability: {probability:.1%}")
    st.write("Input data used by the model:")
    st.dataframe(input_df)
