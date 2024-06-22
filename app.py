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
        st.markdown(f'**{key}**')
        parameters[key] = st.number_input('', key=key.lower().replace(' ', '_'), value=0, format='%f')
with col2:
    for key in list(parameters.keys())[15:]:
        st.markdown(f'**{key}**')
        parameters[key] = st.number_input('', key=key.lower().replace(' ', '_'), value=0, format='%f')

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


