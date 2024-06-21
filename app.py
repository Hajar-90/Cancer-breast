import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load the model
model = load_model('oneclass.h5')

# Function to preprocess the image
def preprocess_image(image):
    size = (224, 224)  # Assuming the model expects 224x224 images
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image = np.asarray(image)
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to make prediction
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

# Streamlit interface
st.title("Image Classification with OneClass Model")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png","pgm"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    prediction = predict(image)
    st.write(f"Prediction: {prediction}")
