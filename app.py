import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import joblib

# Load KNN model and scaler
knn = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load your existing Keras model
model = load_model('oneone.keras')

# Main Streamlit app
st.set_page_config(
    page_title="Breast Cancer Classification",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title('Breast Cancer Classification')

# Sidebar for Mammogram Analysis
uploaded_file = st.sidebar.file_uploader("Upload a Mammogram Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Preprocess the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(image.resize((224, 224))) / 255.0  # Resize and normalize

        # Expand dimensions to create a batch of size 1
        img_array = np.expand_dims(img_array, axis=0)

        # Display the uploaded image
        st.subheader("Uploaded Image")
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make predictions using the loaded model
        prediction = model.predict(img_array)
        prediction_result = 'Malignant' if prediction[0][0] > 0.5 else 'Benign'
        prediction_confidence = prediction[0][0] if prediction_result == 'Malignant' else 1 - prediction[0][0]

        # Display prediction result
        st.write(f'Prediction: {prediction_result}')
        st.write(f'Confidence: {prediction_confidence:.2f}')

    except Exception as e:
        st.error(f"An error occurred: {e}")




