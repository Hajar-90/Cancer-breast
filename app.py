import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load CNN model with error handling
model_loaded = False
try:
    cnn_model = load_model('oneone.keras')
    model_loaded = True
except FileNotFoundError:
    st.error("CNN model file 'oneone.keras' not found. Please upload the model file.")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")

# Main Streamlit app
st.set_page_config(
    page_title="Breast Cancer Classification",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title('Breast Cancer Classification')
uploaded_file = st.file_uploader("Upload a Mammogram Image", type=["jpg", "jpeg", "png", "pgm"])

if uploaded_file is not None and model_loaded:
    try:
        # Load the image using PIL
        image = Image.open(uploaded_file).convert('RGB')  # Convert to RGB
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image for the CNN model
        image_resized = image.resize((224, 224))  # Resize for CNN input
        image_array = np.array(image_resized).reshape((1, 224, 224, 3)) / 255.0  # Normalize

        # Make a prediction using the CNN model
        cnn_prediction = cnn_model.predict(image_array)
        cnn_result = 'Malignant' if cnn_prediction[0][0] > 0.5 else 'Benign'
        cnn_confidence = cnn_prediction[0][0] if cnn_result == 'Malignant' else 1 - cnn_prediction[0][0]
        cnn_confidence *= 100

        # Determine the appropriate emoji based on confidence level
        if cnn_confidence >= 90:
            emoji = '✔️'  # Checkmark for high confidence
        elif cnn_confidence >= 80:
            emoji = '😊'  # Smiling face for good confidence
        elif cnn_confidence >= 70:
            emoji = '😐'  # Neutral face for moderate confidence
        else:
            emoji = '😕'  # Confused face for lower confidence

        # Display the CNN prediction result with styled box
        st.markdown('<div style="background-color:white; padding:10px; border-radius:10px;">'
                    '<p style="color:black; font-size:18px; font-weight:bold;">CNN Prediction</p>'
                    f'<p style="color:black;">Result: {cnn_result}</p>'
                    f'<p style="color:black;">Confidence: {cnn_confidence:.2f}% {emoji}</p>'
                    '</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An unexpected error occurred during image processing or prediction: {e}")
