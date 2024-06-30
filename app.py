import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Specify the absolute path to your model file
cnn_model_path = 'oneone.keras'  # Update with your actual file path

# Load CNN model with error handling
model_loaded = False
try:
    cnn_model = tf.keras.models.load_model(cnn_model_path)
    model_loaded = True
except FileNotFoundError:
    st.error(f"CNN model file '{cnn_model_path}' not found. Please upload the model file.")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")

# Main Streamlit app
st.title('Breast Cancer Classification')
uploaded_file = st.file_uploader("Upload a Mammogram Image", type=["jpg", "jpeg", "png", "pgm"])

if uploaded_file is not None and model_loaded:
    try:
        # Load the image using PIL
        image = Image.open(uploaded_file).convert('RGB')  # Convert to RGB
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image for the CNN model
        image_resized = image.resize((224, 224))  # Resize for CNN input
        image_array = np.array(image_resized)  # Convert to numpy array
        image_array = image_array.reshape((1, 224, 224, 3)) / 255.0  # Normalize

        # Make a prediction using the CNN model
        cnn_prediction = cnn_model.predict(image_array)
        cnn_result = 'Malignant' if cnn_prediction[0][0] > 0.5 else 'Benign'
        cnn_confidence = cnn_prediction[0][0] if cnn_result == 'Malignant' else 1 - cnn_prediction[0][0]
        cnn_confidence *= 100

        # Display the CNN prediction result
        st.write(f'Prediction: {cnn_result}')
        st.write(f'Confidence: {cnn_confidence:.2f}%')

    except Exception as e:
        st.error(f"Error during image processing or prediction: {e}")

