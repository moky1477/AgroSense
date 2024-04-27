import streamlit as st
import pandas as pd
import pickle

# Load the trained model and label encoders
# Load the trained model and label encoders with protocol version 4
with open("Fertilizer-Recommendation-System\Models\model.pkl", "rb") as f:
    model, label_encoders = pickle.load(f)

# Function to preprocess user input
def preprocess_input(user_input):
    # Encode categorical variables
    for column, encoder in label_encoders.items():
        user_input[column] = encoder.transform([user_input[column]])[0]
    # Convert input to DataFrame
    input_df = pd.DataFrame([user_input])
    return input_df

# Function to predict fertilizer recommendation
def predict_fertilizer(user_input):
    input_df = preprocess_input(user_input)
    prediction = model.predict(input_df)
    recommended_fertilizer = label_encoders['Fertilizer Name'].inverse_transform(prediction)[0]
    return recommended_fertilizer

# Streamlit app
st.title("Fertilizer Recommendation System")

# User input fields
temperature = st.number_input("Temperature (°C):", min_value=0, max_value=100, value=25)
humidity = st.number_input("Humidity (%):", min_value=0, max_value=100, value=50)
soil_moisture = st.number_input("Soil Moisture (%):", min_value=0, max_value=100, value=50)
soil_type = st.selectbox("Soil Type:", ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey'])
crop_type = st.selectbox("Crop Type:", ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy', 'Barley', 'Millets', 'Oil seeds', 'Pulses', 'Ground Nuts', 'Wheat'])
nitrogen = st.number_input("Nitrogen (N):", min_value=0, value=0)
potassium = st.number_input("Potassium (P):", min_value=0, value=0)
phosphorous = st.number_input("Phosphorous (K):", min_value=0, value=0)

# Predict button
if st.button("Predict Fertilizer"):
    user_input = {
        'Temperature': temperature,
        'Humidity': humidity,
        'Soil Moisture': soil_moisture,
        'Soil Type': soil_type,
        'Crop Type': crop_type,
        'Nitrogen': nitrogen,
        'Potassium': potassium,
        'Phosphorous': phosphorous
    }
    recommended_fertilizer = predict_fertilizer(user_input)
    st.success(f"The recommended fertilizer for your crop is: {recommended_fertilizer}")
