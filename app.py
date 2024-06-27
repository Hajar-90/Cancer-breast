import streamlit as st
import tensorflow as tf
from keras.preprocessing import image
import numpy as np

# Load the Keras model
model_path = 'oneone.keras'
cnn_model = tf.keras.models.load_model(model_path)

# Function to preprocess image
def preprocess_image(img):
    # Resize image to match model input size
    img = image.load_img(img, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to create batch of 1
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

# Main Streamlit app
st.title('Breast Cancer Classification')
uploaded_file = st.file_uploader("Upload a Mammogram Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess the uploaded image
    img_array = preprocess_image(uploaded_file)
    
    # Make prediction using the CNN model
    prediction = cnn_model.predict(img_array)
    
    # Display the prediction result
    st.write(prediction)
