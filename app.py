import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Define a custom load_model function to handle BatchNormalization deserialization issue
def custom_load_model(filepath):
    from tensorflow.keras.layers import BatchNormalization
    import tensorflow as tf

    # Create a custom object scope that includes BatchNormalization
    custom_objects = {
        'BatchNormalization': BatchNormalization
    }

    with tf.keras.utils.custom_object_scope(custom_objects):
        model = load_model(filepath)

    return model

# Load the trained model with detailed error handling
model_loaded = False
try:
    model = custom_load_model('one.keras')
    model_loaded = True
    st.write("Model loaded successfully.")
except FileNotFoundError:
    st.error("Model file 'oneclasss.keras' not found. Please ensure the file is accessible.")
except TypeError as e:
    st.error(f"TypeError encountered: {e}")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to match the input shape of the model
    img = image.resize((224, 224))
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Normalize the pixel values to be in the range [0, 1]
    img_array = img_array / 255.0
    # Add a batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# Define a function to make predictions on the preprocessed image
def predict_image(image):
    # Preprocess the image
    img = preprocess_image(image)
    # Make predictions using the model
    predictions = model.predict(img)
    # Convert the predicted probabilities to class labels
    predicted_class = 1 if predictions[0][0] > 0.5 else 0
    # Calculate confidence
    confidence = predictions[0][0] * 100 if predicted_class == 1 else (1 - predictions[0][0]) * 100
    # Print the confidence
    st.write(f"Confidence: {confidence:.2f}%")

    return predicted_class

# Streamlit UI
st.title('Breast Cancer Classification')

# File uploader for mammogram image
uploaded_file = st.file_uploader("Upload a Mammogram Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model_loaded:
    # Load the image using PIL
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make predictions on the image
    predicted_class = predict_image(image)

    # Map the predicted class label to the corresponding class name
    label_mapping = {
        0: 'Benign',
        1: 'Malignant'
    }
    predicted_class_name = label_mapping[predicted_class]

    # Print the predicted class name
    st.write("Predicted Class:", predicted_class_name)



