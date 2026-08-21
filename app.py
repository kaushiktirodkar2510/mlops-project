import streamlit as st
from src.predict import predict_price


# Page configuration
st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)


# Title
st.title("🚗 Used Car Price Prediction")
st.write("Enter the details of the used car to estimate its selling price.")


# Input fields
make_year = st.number_input(
    "Manufacturing Year",
    min_value=1990,
    max_value=2026,
    value=2018
)

mileage_kmpl = st.number_input(
    "Mileage (km/l)",
    min_value=1.0,
    max_value=50.0,
    value=15.0
)

engine_cc = st.number_input(
    "Engine Capacity (CC)",
    min_value=500,
    max_value=8000,
    value=1500
)

fuel_type = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "CNG", "Electric", "Hybrid"]
)

owner_count = st.number_input(
    "Number of Previous Owners",
    min_value=0,
    max_value=10,
    value=1
)

brand = st.text_input(
    "Brand",
    value="Toyota"
)

transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic"]
)

color = st.text_input(
    "Color",
    value="White"
)

service_history = st.selectbox(
    "Service History",
    ["Full", "Partial", "None"]
)

accidents_reported = st.number_input(
    "Accidents Reported",
    min_value=0,
    max_value=20,
    value=0
)

insurance_valid = st.selectbox(
    "Insurance Valid",
    ["Yes", "No"]
)


# Prediction button
if st.button("Predict Price"):

    input_data = {
        "make_year": make_year,
        "mileage_kmpl": mileage_kmpl,
        "engine_cc": engine_cc,
        "fuel_type": fuel_type,
        "owner_count": owner_count,
        "brand": brand,
        "transmission": transmission,
        "color": color,
        "service_history": service_history,
        "accidents_reported": accidents_reported,
        "insurance_valid": insurance_valid
    }

    prediction = predict_price(input_data)

    st.success(
        f"Estimated Used Car Price: ${prediction:,.2f}"
    )