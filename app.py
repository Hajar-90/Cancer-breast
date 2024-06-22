import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the CNN model
try:
    cnn_model = tf.keras.models.load_model('oneone.keras')
except (FileNotFoundError, ValueError, TypeError) as e:
    st.error(f"Error loading CNN model: {e}")

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to predict using the CNN model
def predict_image(image_array):
    predictions = cnn_model.predict(image_array)
    result = 'Malignant' if predictions[0][0] > 0.5 else 'Benign'
    confidence = predictions[0][0] if result == 'Malignant' else 1 - predictions[0][0]
    return result, confidence

# Function to highlight gray range
def highlight_gray_range(image_np, gray_lower, gray_upper):
    mask = (image_np >= gray_lower) & (image_np <= gray_upper)
    highlighted_image = np.where(mask, image_np, 0)
    return highlighted_image, mask

# Function to create highlighted overlay
def create_highlighted_overlay(original_image, highlighted_region, mask, highlight_color):
    overlay = np.stack((original_image,) * 3, axis=-1)  # Convert to RGB
    overlay[np.where(mask)] = highlight_color
    return overlay

# Streamlit app
st.title('Breast Cancer Prediction and Image Analysis')

uploaded_file = st.file_uploader("Upload a mammogram image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image_np = np.array(image)

    # Apply the gray range filter and get the mask
    highlighted_image, mask = highlight_gray_range(image_np, 50, 150)  # Example gray range

    # Create the highlighted overlay with a specific color (e.g., red)
    highlight_color = [255, 0, 0]  # Red color for the highlighted overlay
    highlighted_overlay = create_highlighted_overlay(image_np, highlighted_image, mask, highlight_color)

    # Preprocess the image for the CNN model
    image_resized = image.resize((224, 224))  # Resize to the input size the CNN expects
    image_array = np.array(image_resized).reshape((1, 224, 224, 1)) / 255.0  # Normalize the image

    # Predict using the CNN model
    cnn_prediction, cnn_confidence = predict_image(image_array)

    # Display prediction and confidence
    st.subheader('CNN Prediction:')
    if cnn_prediction == 'Malignant':
        st.write(f'**{cnn_prediction}**', ' ', f'({cnn_confidence:.2f})')
    else:
        st.write(f'**{cnn_prediction}**', ' ', f'({cnn_confidence:.2f})')

    # Display the three images in one line horizontally
    st.subheader('Image Analysis:')
    
    # Set up a layout for the images in one row
    col1, col2, col3 = st.beta_columns(3)  # Three columns for each image
    
    # Display original image
    with col1:
        st.image(image_np, caption='Original Image', use_column_width=True, channels='GRAY')
    
    # Display highlighted image
    with col2:
        st.image(highlighted_image, caption='Highlighted Image', use_column_width=True, channels='GRAY')

    # Display highlighted overlay
    with col3:
        st.image(highlighted_overlay, caption='Highlighted Overlay', use_column_width=True)

    # Optionally, display additional information or analysis results here





