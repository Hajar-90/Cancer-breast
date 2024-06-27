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
uploaded_file = st.sidebar.file_uploader("Upload a Mammogram Image", type=["jpg", "jpeg", "png", "pgm"])

# Display uploaded image and processing
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

        # Display images based on user selection with specified width
        st.subheader("Uploaded Image")
        if show_original:
            st.image(image_resized, caption='Original Image', width=500, channels='L')

        if show_highlighted:
            st.image(highlighted_image, caption='Highlighted Image', width=500, channels='L')

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

        # Display CNN prediction for the uploaded image
        if model_loaded:
            # Preprocess the image for the CNN model
            image_rgb = image.convert('RGB')  # Convert to RGB
            image_resized_cnn = image_rgb.resize((224, 224))  # Resize for CNN input
            image_array = np.array(image_resized_cnn).reshape((1, 224, 224, 3)) / 255.0  # Normalize

            # Make a prediction using the CNN model
            cnn_prediction = cnn_model.predict(image_array)
            cnn_result = 'Malignant' if cnn_prediction[0][0] > 0.5 else 'Benign'
            cnn_confidence = cnn_prediction[0][0] if cnn_result == 'Malignant' else 1 - cnn_prediction[0][0]
            cnn_confidence *= 100

            # Determine the appropriate emoji based on confidence level
            if cnn_confidence >= 90:
                emoji = '✔️'  # Checkmark for high confidence
            elif cnn_confidence >= 80:
                emoji = '😊'  # Smiling face for good confidence
            elif cnn_confidence >= 70:
                emoji = '😐'  # Neutral face for moderate confidence
            else:
                emoji = '😕'  # Confused face for lower confidence

            # Display the CNN prediction result with styled box
            st.markdown('<div style="background-color:white; padding:10px; border-radius:10px;">'
                        '<p style="color:black; font-size:18px; font-weight:bold;">CNN Prediction</p>'
                        f'<p style="color:black;">Result: {cnn_result}</p>'
                        f'<p style="color:black;">Confidence: {cnn_confidence:.2f}% {emoji}</p>'
                        '</div>', unsafe_allow_html=True)

    except ValueError as e:
        st.sidebar.error(f"ValueError: {e}")
    except Exception as e:
        st.sidebar.error(f"An unexpected error occurred during image processing or prediction: {e}")

# Main Section for Breast Cancer Prediction Parameters Input
st.title('Breast Cancer Prediction Parameters Input')

# Information about each parameter (tooltips or descriptions)
parameter_info = {
    'Mean Radius': 'Mean radius of the cells.',
    'Mean Texture': 'Mean texture of the cells.',
    'Mean Perimeter': 'Mean perimeter of the cells.',
    'Mean Area': 'Mean area of the cells.',
    'Mean Smoothness': 'Mean smoothness of the cells.',
    'Mean Compactness': 'Mean compactness of the cells.',
    'Mean Concavity': 'Mean concavity of the cells.',
    'Mean Concave Points': 'Mean number of concave portions of the contour.',
    'Mean Symmetry': 'Mean symmetry of the cells.',
    'Mean Fractal Dimension': 'Mean fractal dimension of the cells.',
    'Radius Error': 'Standard error of the radius of the cells.',
    'Texture Error': 'Standard error of the texture of the cells.',
    'Perimeter Error': 'Standard error of the perimeter of the cells.',
    'Area Error': 'Standard error of the area of the cells.',
    'Smoothness Error': 'Standard error of the smoothness of the cells.',
    'Compactness Error': 'Standard error of the compactness of the cells.',
    'Concavity Error': 'Standard error of the concavity of the cells.',
    'Concave Points Error': 'Standard error of the number of concave portions of the contour.',
    'Symmetry Error': 'Standard error of the symmetry of the cells.',
    'Fractal Dimension Error': 'Standard error of the fractal dimension of the cells.',
    'Worst Radius': 'Worst (largest) radius of the cells.',
    'Worst Texture': 'Worst (most severe) texture of the cells.',
    'Worst Perimeter': 'Worst (largest) perimeter of the cells.',
    'Worst Area': 'Worst (largest) area of the cells.',
    'Worst Smoothness': 'Worst (most severe) smoothness of the cells.',
    'Worst Compactness': 'Worst (most severe) compactness of the cells.',
    'Worst Concavity': 'Worst (most severe) concavity of the cells.',
    'Worst Concave Points': 'Worst (most severe) number of concave portions of the contour.',
    'Worst Symmetry': 'Worst (most severe) symmetry of the cells.',
    'Worst Fractal Dimension': 'Worst (most severe) fractal dimension of the cells.'
}

# Layout with columns for text inputs
col1, col2 = st.columns(2)

# Define text inputs for parameters with smaller font size and tooltips/descriptions
parameters = {}
with col1:
    for key in list(parameter_info.keys())[:15]:
        parameters[key] = st.text_input(key, key=key.lower().replace(' ', '_'), value='0', max_chars=10,
                                        help=f"{parameter_info[key]}")
with col2:
    for key in list(parameter_info.keys())[15:]:
        parameters[key] = st.text_input(key, key=key.lower().replace(' ', '_'), value='0', max_chars=10,
                                        help=f"{parameter_info[key]}")

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
