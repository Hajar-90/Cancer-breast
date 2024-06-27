import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Absolute path to the .keras file
model_path = 'oneone.keras'

# Load Keras model
@st.cache(allow_output_mutation=True)
def load_model():
    return tf.keras.models.load_model(model_path)

model = load_model()

# Function to preprocess the image
def preprocess_image(image):
    # Resize image to match model input size
    image = image.resize((224, 224))
    # Convert image to numpy array
    image = np.asarray(image)
    # Normalize pixel values (assuming model expects inputs in range [0, 1])
    image = image / 255.0
    # Expand dimensions to create a batch of size 1
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit app interface
st.title('Upload Image for Prediction')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess image for prediction
    processed_image = preprocess_image(image)
    
    # Make prediction using the model
    prediction = model.predict(processed_image)
    
    # Display prediction result
    st.write(f'Prediction: {prediction}')

