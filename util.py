import base64
import io
import streamlit as st
from PIL import ImageOps, Image
import numpy as np


def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)


def classify(image, model, class_names):
    """
    This function takes an image, a model, and a list of class names and returns the predicted class 
    and confidence score of the image.

    Parameters:
        image (PIL.Image.Image): The input image as a PIL Image object.
        model (tensorflow.keras.Model): A trained machine learning model for image classification.
        class_names (list): A list of class names corresponding to the classes that the model can predict.

    Returns:
        A tuple of the predicted class name and the confidence score for that prediction.
    """
    # Convert the image to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')

    # Resize and preprocess the image
    img = image.resize((64, 64))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)

    # Make prediction
    predictions = model.predict(img)
    
    # Extract predicted class index
    predicted_class_index = np.argmax(predictions[0])
    label_mapping = {
    0: 'Malignant',
    1: 'Benign'
    }

# Get the predicted class name using the label mapping
    predicted_class_name = label_mapping[predicted_class_index]
    # Get class name and confidence score

    return predicted_class_name


