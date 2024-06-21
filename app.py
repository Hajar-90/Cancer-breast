import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D
from PIL import Image, ImageOps
import numpy as np

# Custom layer handling for BatchNormalization
def custom_batchnorm_from_config(config):
    axis = config.pop('axis', -1)
    momentum = config.pop('momentum', 0.99)
    epsilon = config.pop('epsilon', 1e-3)
    beta_initializer = tf.keras.initializers.deserialize(config.pop('beta_initializer'))
    gamma_initializer = tf.keras.initializers.deserialize(config.pop('gamma_initializer'))
    moving_mean_initializer = tf.keras.initializers.deserialize(config.pop('moving_mean_initializer'))
    moving_variance_initializer = tf.keras.initializers.deserialize(config.pop('moving_variance_initializer'))
    
    layer = BatchNormalization(
        axis=axis,
        momentum=momentum,
        epsilon=epsilon,
        beta_initializer=beta_initializer,
        gamma_initializer=gamma_initializer,
        moving_mean_initializer=moving_mean_initializer,
        moving_variance_initializer=moving_variance_initializer,
        **config
    )
    
    return layer

# Custom object dictionary
custom_objects = {
    'InputLayer': InputLayer,
    'Conv2D': Conv2D,
    'BatchNormalization': custom_batchnorm_from_config,
    'Activation': Activation,
    'GlobalAveragePooling2D': GlobalAveragePooling2D
}

# Function to load the model with custom objects
def load_model_safely(model_path):
    try:
        model = load_model(model_path, custom_objects=custom_objects)
        st.success("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
model_path = 'oneclass.h5'
model = load_model_safely(model_path)

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
    if model is not None:
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        return prediction
    else:
        st.error("Model is not loaded properly.")
        return None

# Streamlit interface
st.title("Image Classification with OneClass Model")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    prediction = predict(image)
    if prediction is not None:
        st.write(f"Prediction: {prediction}")
    else:
        st.error("Prediction could not be made due to an error.")

# To run the streamlit app, use the following command in your terminal:
# streamlit run your_script_name.py


