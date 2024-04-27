from dotenv import load_dotenv
import streamlit as st
import os
import numpy as np
import joblib
import pickle
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input_prompt, input_text):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content([input_text, input_prompt])
    return response.text

def load_ml_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

svm_model = load_ml_model('Crop-Recommendation-System/Models/SVMClassifier.pkl')
naive_bayes_model = load_ml_model('Crop-Recommendation-System/Models/NBClassifier.pkl')
dl_model = load_ml_model('Crop-Recommendation-System/Models/SVMClassifier.pkl')

# Streamlit UI
def main():
    st.title("Crop Recommendation System")

    # User Inputs
    N = st.number_input('Nitrogen', min_value=0)
    P = st.number_input('Phosphorus', min_value=0)
    K = st.number_input('Potassium', min_value=0)
    temperature = st.number_input('Temperature', min_value=-10.0)
    humidity = st.number_input('Humidity', min_value=0.0, max_value=100.0)
    ph = st.number_input('pH Level', min_value=0.0, max_value=14.0)
    rainfall = st.number_input('Rainfall', min_value=0.0)

    # Model selection
    model_option = st.selectbox("Select Model for Prediction", 
                                ['Naive Bayes', 'SVM', 'Neural Network'])

    # Predict button
    if st.button('Predict'):
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        
        if model_option == 'Naive Bayes':
            prediction = naive_bayes_model.predict(input_data)
        elif model_option == 'SVM':
            prediction = svm_model.predict(input_data)
        elif model_option == 'Neural Network':
            prediction = dl_model.predict(input_data)
        
        # The code for getting crop recommendation using Gemini AI
        crop_name = prediction[0]
        input_prompt = """
        As an experienced farmer with in-depth knowledge of various crops, your expertise is sought by a beginner farmer looking to cultivate a specific crop. Provide detailed guidance on the chosen crop, covering the following aspects:

        1. Briefly outline the basic steps, advantages, and applications of the selected crop to help the user understand its significance in farming.
        2. Share specific techniques tailored for a beginner farmer to successfully grow the mentioned crop.
        3. Ensure the information provided is accurate, avoiding any misleading or incorrect advice.

        Your goal is to empower the user with valuable insights and practical tips to foster a successful farming experience.
        """
        response = get_gemini_response(input_prompt, crop_name)
        
        st.success(f"The recommended crop is: {crop_name}")
        st.subheader("More about the recommended crop:")
        st.write(response)


if __name__ == "__main__":
    main()
