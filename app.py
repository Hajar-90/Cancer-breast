import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from util import classify, set_background
import joblib
knn = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')
# Function to highlight the gray range
def highlight_gray_range(image_np, gray_lower, gray_upper):
    mask = (image_np >= gray_lower) & (image_np <= gray_upper)
    highlighted_image = np.where(mask, image_np, 0)
    return highlighted_image, mask

# Function to create the highlighted overlay
def create_highlighted_overlay(original_image, highlighted_region, mask, highlight_color):
    overlay = np.stack((original_image,) * 3, axis=-1)  # Convert to RGB
    overlay[np.where(mask)] = highlight_color
    return overlay

# Main streamlit app
st.title('Mammogram Gray Range Highlighter')

# Sidebar inputs for gray range
st.sidebar.header('Select Gray Range')
gray_lower = st.sidebar.slider('Lower Bound of Gray Range', 0, 255, 50)
gray_upper = st.sidebar.slider('Upper Bound of Gray Range', 0, 255, 150)

# File uploader for mammogram image
uploaded_file = st.file_uploader("Upload a Mammogram Image", type=["jpg", "jpeg", "png", "pgm"])

if uploaded_file is not None:
    # Load the image using PIL
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image_np = np.array(image)

    # Apply the gray range filter and get the mask
    highlighted_image, mask = highlight_gray_range(image_np, gray_lower, gray_upper)

    # Create the highlighted overlay with a specific color (e.g., red)
    highlight_color = [255, 0, 0]  # Red color for the highlighted overlay
    highlighted_overlay = create_highlighted_overlay(image_np, highlighted_image, mask, highlight_color)

    # Display the original image
    st.image(image_np, caption='Original Image', use_column_width=True, channels='GRAY')

    # Display the highlighted image
    st.image(highlighted_image, caption='Highlighted Image', use_column_width=True, channels='GRAY')

    # Display the highlighted overlay
    st.image(highlighted_overlay, caption='Highlighted Overlay', use_column_width=True)

    # Plot the mask and the highlighted overlay
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(mask, cmap='gray')
    axs[0].set_title('Mask')
    axs[0].axis('off')

    axs[1].imshow(highlighted_overlay)
    axs[1].set_title('Highlighted Overlay')
    axs[1].axis('off')

    # Show the plot
    st.pyplot(fig)

set_background('bgs/bg5.jpg')

# set title
st.title('Breast Cancer classification')


st.title('Breast Cancer Prediction Parameters Input')

# Create text inputs for each parameter
mean_radius = st.text_input('Mean Radius')
mean_texture = st.text_input('Mean Texture')
mean_perimeter = st.text_input('Mean Perimeter')
mean_area = st.text_input('Mean Area')
mean_smoothness = st.text_input('Mean Smoothness')
mean_compactness = st.text_input('Mean Compactness')
mean_concavity = st.text_input('Mean Concavity')
mean_concave_points = st.text_input('Mean Concave Points')
mean_symmetry = st.text_input('Mean Symmetry')
mean_fractal_dimension = st.text_input('Mean Fractal Dimension')
radius_error = st.text_input('Radius Error')
texture_error = st.text_input('Texture Error')
perimeter_error = st.text_input('Perimeter Error')
area_error = st.text_input('Area Error')
smoothness_error = st.text_input('Smoothness Error')
compactness_error = st.text_input('Compactness Error')
concavity_error = st.text_input('Concavity Error')
concave_points_error = st.text_input('Concave Points Error')
symmetry_error = st.text_input('Symmetry Error')
fractal_dimension_error = st.text_input('Fractal Dimension Error')
worst_radius = st.text_input('Worst Radius')
worst_texture = st.text_input('Worst Texture')
worst_perimeter = st.text_input('Worst Perimeter')
worst_area = st.text_input('Worst Area')
worst_smoothness = st.text_input('Worst Smoothness')
worst_compactness = st.text_input('Worst Compactness')
worst_concavity = st.text_input('Worst Concavity')
worst_concave_points = st.text_input('Worst Concave Points')
worst_symmetry = st.text_input('Worst Symmetry')
worst_fractal_dimension = st.text_input('Worst Fractal Dimension')

# Add a button to submit the data
if st.button('Predict'):
    # Collect the entered data
    data = np.array([
        mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, 
        mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, 
        mean_fractal_dimension, radius_error, texture_error, perimeter_error, 
        area_error, smoothness_error, compactness_error, concavity_error, 
        concave_points_error, symmetry_error, fractal_dimension_error, worst_radius, 
        worst_texture, worst_perimeter, worst_area, worst_smoothness, worst_compactness, 
        worst_concavity, worst_concave_points, worst_symmetry, worst_fractal_dimension
    ], dtype=float).reshape(1, -1)
    
    # Scale the input data
    data_scaled = scaler.transform(data)
    
    # Make a prediction
    prediction = knn.predict(data_scaled)
    prediction_proba = knn.predict_proba(data_scaled)
    
    # Display the result
    result = 'Malignant' if prediction[0] == 1 else 'Benign'
    st.write(f'Prediction: {result}')
    st.write(f'Prediction Probability: {prediction_proba[0]}')
