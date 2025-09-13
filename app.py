import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt 

# Load model
model = joblib.load("random_forest_model.joblib")
scaler = joblib.load("scaler.joblib") 

# Streamlit App 
st.title(" Climate Disaster Risk Prediction - Vietnam")
st.markdown("This app predicts disaster risk levels based on historical climate data (1901–2020).")

# Sidebar inputs
st.sidebar.header("Input Climate Features")

min_temp = st.sidebar.number_input("Minimum Temperature (°C)", value=20.0)
mean_temp = st.sidebar.number_input("Mean Temperature (°C)", value=25.0)
max_temp = st.sidebar.number_input("Maximum Temperature (°C)", value=30.0)
precipitation = st.sidebar.number_input("Precipitation (mm)", value=100.0)
month = st.sidebar.slider("Month (1-12)", 1, 12, 6)
year = st.sidebar.number_input("Year", value=2000, min_value=1901, max_value=2025)

# Prepare input for prediction
input_data = pd.DataFrame({
    "MinTemperature": [min_temp],
    "MeanTemperature": [mean_temp],
    "MaxTemperature": [max_temp],
    "Precipitation": [precipitation],
    "Month": [month],
    "Year": [year]
})

# Make sure input columns match scaler
scaler_features = scaler.feature_names_in_  
input_for_scaler = input_data[scaler_features]

scaled_input = scaler.transform(input_for_scaler)

# Make sure input columns match model
model_features = model.feature_names_in_ 
input_for_model = pd.DataFrame(scaled_input, columns=scaler_features)
for col in model_features:
    if col not in input_for_model.columns:
        input_for_model[col] = input_data[col] 

# Reorder columns exactly for model
input_for_model = input_for_model[model_features]

# Predict
if st.button("Predict Risk Level"):
    prediction = model.predict(input_for_model)[0]
    prediction = max(1, min(8, prediction))  # Force into 1–8 range
    st.success(f" Predicted Risk Level: **{prediction}**")
    st.info("Levels: 1 = Low Risk → 8 = High Risk")


