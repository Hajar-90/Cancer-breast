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
    st.error("CNN model file 'model.keras' not found. Please upload the model file.")
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
with st.sidebar:
    st.markdown('## Mammogram Analysis')
    uploaded_file = st.file_uploader("Upload a Mammogram Image", type=["jpg", "jpeg", "png", "pgm"])
    if uploaded_file is not None:
        st.markdown('### Select Gray Range')
        gray_lower = st.slider('Lower Bound of Gray Range', min_value=0, max_value=255, value=50, step=1, format='%d')
        gray_upper = st.slider('Upper Bound of Gray Range', min_value=0, max_value=255, value=150, step=1, format='%d')

        try:
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

            if model_loaded:
                # Preprocess the image for the CNN model
                image_rgb = image.convert('RGB')  # Convert to RGB
                image_resized = image_rgb.resize((224, 224))  # Resize to the input size the CNN expects
                image_array = np.array(image_resized).reshape((1, 224, 224, 3)) / 255.0  # Normalize the image

                # Make a prediction using the CNN model
                cnn_prediction = cnn_model.predict(image_array)
                cnn_result = 'Malignant' if cnn_prediction[0][0] > 0.5 else 'Benign'
                cnn_confidence = cnn_prediction[0][0] if cnn_result == 'Malignant' else 1 - cnn_prediction[0][0]

                # Display the CNN prediction result
                st.subheader('CNN Prediction')
                st.markdown(f'**Result**: {cnn_result}')
                st.markdown(f'**Confidence**: {cnn_confidence:.2f}')

        except ValueError as e:
            st.error(f"ValueError: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred during image processing or prediction: {e}")

# Main Section for Breast Cancer Prediction Parameters
st.title('Breast Cancer Prediction Parameters Input')

# Layout with columns for text inputs
col1, col2 = st.columns(2)

# Define text inputs for parameters
parameters_left = {
    'Mean Radius': st.text_input('Mean Radius'),
    'Mean Texture': st.text_input('Mean Texture'),
    'Mean Perimeter': st.text_input('Mean Perimeter'),
    'Mean Area': st.text_input('Mean Area'),
    'Mean Smoothness': st.text_input('Mean Smoothness'),
    'Mean Compactness': st.text_input('Mean Compactness'),
    'Mean Concavity': st.text_input('Mean Concavity'),
    'Mean Concave Points': st.text_input('Mean Concave Points'),
    'Mean Symmetry': st.text_input('Mean Symmetry'),
    'Mean Fractal Dimension': st.text_input('Mean Fractal Dimension')
}

parameters_right = {
    'Radius Error': st.text_input('Radius Error'),
    'Texture Error': st.text_input('Texture Error'),
    'Perimeter Error': st.text_input('Perimeter Error'),
    'Area Error': st.text_input('Area Error'),
    'Smoothness Error': st.text_input('Smoothness Error'),
    'Compactness Error': st.text_input('Compactness Error'),
    'Concavity Error': st.text_input('Concavity Error'),
    'Concave Points Error': st.text_input('Concave Points Error'),
    'Symmetry Error': st.text_input('Symmetry Error'),
    'Fractal Dimension Error': st.text_input('Fractal Dimension Error')
}

# Predict button
if st.button('Predict'):
    try:
        # Collect the entered data
        data = np.array([
            list(parameters_left.values()),
            list(parameters_right.values())
        ], dtype=float).reshape(1, -1)

        # Scale the input data
        data_scaled = scaler.transform(data)

        # Make a prediction
        prediction = knn.predict(data_scaled)
        prediction_proba = knn.predict_proba(data_scaled)

        # Display the result
        result = 'Malignant' if prediction[0] == 1 else 'Benign'
        st.subheader('KNN Prediction')
        st.markdown(f'**Result**: {result}')
        st.markdown(f'**Probability**: {prediction_proba[0][1]:.2f}')

    except ValueError as e:
        st.error(f"ValueError: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred during prediction: {e}")

