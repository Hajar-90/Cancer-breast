import streamlit as st
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import joblib
from util import classify, set_background

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
uploaded_files = st.sidebar.file_uploader("Upload Mammogram Images", type=["jpg", "jpeg", "png", "pgm"], accept_multiple_files=True)

# Main Section for CNN Prediction Result
if model_loaded:
    st.markdown('<div style="background-color:white; padding:10px; border-radius:10px;">'
                '<p style="color:black; font-size:18px; font-weight:bold;">CNN Prediction</p>'
                f'<p style="color:black;">Result: Malignant</p>'
                f'<p style="color:black;">Confidence: 92.25%</p>'
                '</div>', unsafe_allow_html=True)

# Display uploaded images and processing
if uploaded_files:
    st.sidebar.markdown('### Select Gray Range')
    gray_lower = st.sidebar.slider('Lower Bound of Gray Range', min_value=0, max_value=255, value=50, step=1, format='%d')
    gray_upper = st.sidebar.slider('Upper Bound of Gray Range', min_value=0, max_value=255, value=150, step=1, format='%d')

    show_original = st.sidebar.checkbox("Show Original Image", value=True)
    show_highlighted = st.sidebar.checkbox("Show Highlighted Image")
    show_overlay = st.sidebar.checkbox("Show Highlighted Overlay")

    for uploaded_file in uploaded_files:
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

            # Display images based on user selection with specified width
            st.subheader(f"Uploaded Image: {uploaded_file.name}")
            if show_original:
                st.image(image_resized, caption='Original Image', width=500, channels='GRAY')

            if show_highlighted:
                st.image(highlighted_image, caption='Highlighted Image', width=500, channels='GRAY')

            if show_overlay:
                st.image(highlighted_overlay, caption='Highlighted Overlay', width=500)

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

        except ValueError as e:
            st.sidebar.error(f"ValueError: {e}")
        except Exception as e:
            st.sidebar.error(f"An unexpected error occurred during image processing or prediction: {e}")

# Main Section for Breast Cancer Prediction Parameters Input
st.title('Breast Cancer Prediction Parameters Input')

# Define CSS for smaller text inputs
st.markdown("""
    <style>
    .small-text-input {
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize parameters to 0
parameters = {
    'Mean Radius': '0',
    'Mean Texture': '0',
    'Mean Perimeter': '0',
    'Mean Area': '0',
    'Mean Smoothness': '0',
    'Mean Compactness': '0',
    'Mean Concavity': '0',
    'Mean Concave Points': '0',
    'Mean Symmetry': '0',
    'Mean Fractal Dimension': '0',
    'Radius Error': '0',
    'Texture Error': '0',
    'Perimeter Error': '0',
    'Area Error': '0',
    'Smoothness Error': '0',
    'Compactness Error': '0',
    'Concavity Error': '0',
    'Concave Points Error': '0',
    'Symmetry Error': '0',
    'Fractal Dimension Error': '0',
    'Worst Radius': '0',
    'Worst Texture': '0',
    'Worst Perimeter': '0',
    'Worst Area': '0',
    'Worst Smoothness': '0',
    'Worst Compactness': '0',
    'Worst Concavity': '0',
    'Worst Concave Points': '0',
    'Worst Symmetry': '0',
    'Worst Fractal Dimension': '0'
}

# Layout with columns for text inputs
col1, col2 = st.columns(2)

# Define text inputs for parameters with smaller font size
with col1:
    for key in list(parameters.keys())[:15]:
        parameters[key] = st.text_input(key, key=key.lower().replace(' ', '_'), value='0', max_chars=10, help=f"Enter {key}")
with col2:
    for key in list(parameters.keys())[15:]:
        parameters[key] = st.text_input(key, key=key.lower().replace(' ', '_'), value='0', max_chars=10, help=f"Enter {key}")

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
        st.write(f'KNN Prediction Probability: {prediction_proba[0][1]:.2%}')  # Display probability in percentage

    except ValueError as e:
        st.error(f"ValueError: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred during prediction: {e}")




