import streamlit as st
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import joblib
from util import highlight_gray_range, create_highlighted_overlay, set_background

# Load KNN model and scaler
knn = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load CNN model with error handling
model_loaded = False
try:
    cnn_model = tf.keras.models.load_model('oneone.keras')
    model_loaded = True
except FileNotFoundError:
    st.error("CNN model file 'oneone.keras' not found. Please upload the model file.")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")

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
st.set_page_config(
    page_title="Breast Cancer Classification",
    layout="wide",
    initial_sidebar_state="expanded"
)
set_background('bgs/bg5.jpg')

# Title and Sidebar for Mammogram Analysis
st.title('Breast Cancer Classification')
uploaded_file = st.sidebar.file_uploader("Upload a Mammogram Image", type=["jpg", "jpeg", "png", "pgm"])

if uploaded_file is not None:
    st.sidebar.markdown('### Select Gray Range')
    gray_lower = st.sidebar.slider('Lower Bound of Gray Range', min_value=0, max_value=255, value=50, step=1, format='%d')
    gray_upper = st.sidebar.slider('Upper Bound of Gray Range', min_value=0, max_value=255, value=150, step=1, format='%d')

    show_original = st.sidebar.checkbox("Show Original Image", value=True)
    show_highlighted = st.sidebar.checkbox("Show Highlighted Image")
    show_overlay = st.sidebar.checkbox("Show Highlighted Overlay")

    try:
        # Load the image using PIL
        image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
        image_np = np.array(image)

        # Resize image to fit display
        image_resized = image.resize((500, 500))

        # Apply the gray range filter and get the mask
        highlighted_image, mask = highlight_gray_range(image_np, gray_lower, gray_upper)

        # Create the highlighted overlay with a specific color (e.g., red)
        highlight_color = [255, 0, 0]  # Red color for the highlighted overlay
        highlighted_overlay = create_highlighted_overlay(image_np, highlighted_image, mask, highlight_color)

        # Display images based on user selection
        if show_original:
            st.image(image_resized, caption='Original Image', use_column_width=True, channels='GRAY')
        
        if show_highlighted:
            st.image(highlighted_image, caption='Highlighted Image', use_column_width=True, channels='GRAY')
        
        if show_overlay:
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

        if model_loaded:
            # Preprocess the image for the CNN model
            image_rgb = image.convert('RGB')  # Convert to RGB
            image_resized_cnn = image_rgb.resize((224, 224))  # Resize for CNN input
            image_array = np.array(image_resized_cnn).reshape((1, 224, 224, 3)) / 255.0  # Normalize

            # Make a prediction using the CNN model
            cnn_prediction = cnn_model.predict(image_array)
            cnn_result = 'Malignant' if cnn_prediction[0][0] > 0.5 else 'Benign'
            cnn_confidence = cnn_prediction[0][0] if cnn_result == 'Malignant' else 1 - cnn_prediction[0][0]
            cnn_confidence_percentage = cnn_confidence * 100  # Convert confidence to percentage

            # Display the CNN prediction result in a prominent way
            st.subheader('CNN Prediction')
            result_html = f'<p style="font-size: 24px; font-weight: bold;">Result: {cnn_result}</p>'
            confidence_html = f'<p style="font-size: 20px;">Confidence: {cnn_confidence_percentage:.2f}%</p>'
            st.markdown(result_html, unsafe_allow_html=True)
            st.markdown(confidence_html, unsafe_allow_html=True)

    except ValueError as e:
        st.sidebar.error(f"ValueError: {e}")
    except Exception as e:
        st.sidebar.error(f"An unexpected error occurred during image processing or prediction: {e}")

# Main Section for Breast Cancer Prediction Parameters Input
st.title('Breast Cancer Prediction Parameters Input')

# Information about each parameter
parameters_info = {
    'Mean Radius': 'Mean radius of the cell nuclei (mm)',
    'Mean Texture': 'Mean texture of the cell nuclei (unitless)',
    'Mean Perimeter': 'Mean perimeter of the cell nuclei (mm)',
    'Mean Area': 'Mean area of the cell nuclei (mm^2)',
    'Mean Smoothness': 'Mean smoothness of the cell nuclei (unitless)',
    'Mean Compactness': 'Mean compactness of the cell nuclei (unitless)',
    'Mean Concavity': 'Mean concavity of the cell nuclei (unitless)',
    'Mean Concave Points': 'Mean number of concave portions of the contour of the cell nuclei (unitless)',
    'Mean Symmetry': 'Mean symmetry of the cell nuclei (unitless)',
    'Mean Fractal Dimension': 'Mean fractal dimension of the cell nuclei (unitless)',
    'Radius Error': 'Standard error of the radius of the cell nuclei (mm)',
    'Texture Error': 'Standard error of the texture of the cell nuclei (unitless)',
    'Perimeter Error': 'Standard error of the perimeter of the cell nuclei (mm)',
    'Area Error': 'Standard error of the area of the cell nuclei (mm^2)',
    'Smoothness Error': 'Standard error of the smoothness of the cell nuclei (unitless)',
    'Compactness Error': 'Standard error of the compactness of the cell nuclei (unitless)',
    'Concavity Error': 'Standard error of the concavity of the cell nuclei (unitless)',
    'Concave Points Error': 'Standard error of the concave points of the cell nuclei (unitless)',
    'Symmetry Error': 'Standard error of the symmetry of the cell nuclei (unitless)',
    'Fractal Dimension Error': 'Standard error of the fractal dimension of the cell nuclei (unitless)',
    'Worst Radius': 'Worst (largest) radius of the cell nuclei (mm)',
    'Worst Texture': 'Worst (highest) texture of the cell nuclei (unitless)',
    'Worst Perimeter': 'Worst (largest) perimeter of the cell nuclei (mm)',
    'Worst Area': 'Worst (largest) area of the cell nuclei (mm^2)',
    'Worst Smoothness': 'Worst (highest) smoothness of the cell nuclei (unitless)',
    'Worst Compactness': 'Worst (highest) compactness of the cell nuclei (unitless)',
    'Worst Concavity': 'Worst (highest) concavity of the cell nuclei (unitless)',
    'Worst Concave Points': 'Worst (highest) number of concave portions of the contour of the cell nuclei (unitless)',
    'Worst Symmetry': 'Worst (highest) symmetry of the cell nuclei (unitless)',
    'Worst Fractal Dimension': 'Worst (highest) fractal dimension of the cell nuclei (unitless)'
}

# Layout with columns for text inputs
col1, col2 = st.columns(2)

# Define text inputs for parameters with smaller font size and include tooltips
with col1:
    for key in list(parameters_info.keys())[:15]:
        st.text_area(key, key=key.lower().replace(' ', '_'), value='0', max_chars=10, height=60, help=parameters_info[key])
with col2:
    for key in list(parameters_info.keys())[15:]:
        st.text_area(key, key=key.lower().replace(' ', '_'), value='0', max_chars=10, height=60, help=parameters_info[key])

# Predict button
if st.button('Predict'):
    try:
        # Collect the entered data
        data = np.array(list(parameters.values()), dtype=float).reshape(1, -1)

        # Scale the input data
        data_scaled = scaler.transform(data)

        # Make a prediction
        prediction = knn.predict(data_scaled)
        prediction_proba = knn.predict_proba(data_scaled)

        # Display the result
        result = 'Malignant' if prediction[0] == 1 else 'Benign'
        st.write(f'KNN Prediction: {result}')
        st.write(f'KNN Prediction Probability: {prediction_proba[0][1]:.2f}')

    except ValueError as e:
        st.error(f"ValueError: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred during prediction: {e}")


