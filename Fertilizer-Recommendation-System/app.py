import streamlit as st
import pandas as pd
import pickle
import os

from dotenv import load_dotenv
import numpy as np
import pickle
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input_prompt, input_text):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content([input_text, input_prompt])
    return response.text

# Load the pickled model
with open('Fertilizer-Recommendation-System\models.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the pickled label encoder for fertilizer
with open('Fertilizer-Recommendation-System\model_fertilizer.pkl', 'rb') as f:
    encode_ferti = pickle.load(f)

# Function to make predictions
def predict_fertilizer(temperature, humidity, moisture, soil_type, crop_type, nitrogen, phosphorous, potassium):
    prediction = model.predict([[temperature, humidity, moisture, soil_type, crop_type, nitrogen, phosphorous, potassium]])
    # Decode the predicted label using the label encoder
    predicted_fertilizer = encode_ferti.inverse_transform(prediction)[0]
    return predicted_fertilizer

# Streamlit app
def main():
    st.title("Fertilizer Recommendation System")
    
    # User input fields
    temperature = st.slider("Temperature", min_value=0, max_value=100, step=1)
    humidity = st.slider("Humidity", min_value=0, max_value=100, step=1)
    moisture = st.slider("Moisture", min_value=0, max_value=100, step=1)
    soil_type = st.selectbox("Soil Type", ["Black", "Clay", "Red", "Silt"])
    crop_type = st.selectbox("Crop Type", ["Wheat", "Rice", "Maize", "Barley"])
    nitrogen = st.slider("Nitrogen", min_value=0, max_value=100, step=1)
    phosphorous = st.slider("Phosphorous", min_value=0, max_value=100, step=1)
    potassium = st.slider("Potassium", min_value=0, max_value=100, step=1)
    
    if st.button("Predict"):
        # Convert categorical variables to numerical
        soil_type_dict = {"Black": 0, "Clay": 1, "Red": 2, "Silt": 3}
        crop_type_dict = {"Wheat": 0, "Rice": 1, "Maize": 2, "Barley": 3}
        soil_type_num = soil_type_dict[soil_type]
        crop_type_num = crop_type_dict[crop_type]
        
        # Predict fertilizer
        prediction = predict_fertilizer(temperature, humidity, moisture, soil_type_num, crop_type_num, nitrogen, phosphorous, potassium)
        
        # st.success(f"The recommended fertilizer for your input is: {prediction}")

        # The code for getting crop recommendation using Gemini AI
        fert_name = prediction
        input_prompt = """
        As an experienced agronomist familiar with various crops and their nutrient requirements, your expertise is sought by a farmer seeking optimal fertilizer recommendations.

        You are given the name of a fertilizer which is being predicted to be used by the user, your job is to tell the user more about the fertilizer,
        cover these topics:
        1. How to use the given fertilizer
        2. Side Effects of the given fertilizer
        3. Advantages and Disadvantages of the given fertilizer

        Your job is to give highly accurate information about this.
        
        """
        response = get_gemini_response(input_prompt, fert_name)
        
        st.success(f"The recommended fertilizer is: {fert_name}")
        st.subheader("More about the recommended Fertilizer:")
        st.write(response)

if __name__ == "__main__":
    main()
