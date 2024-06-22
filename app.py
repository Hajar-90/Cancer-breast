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

# Load CNN model with detailed error handling
model_loaded = False
try:
    cnn_model = tf.keras.models.load_model('oneone.keras')
    model_loaded = True
except FileNotFoundError:
    st.error("CNN model file 'model.keras' not found. Please upload the model file.")
except TypeError as e:
    st.error(f"TypeError encountered: {e}")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")

# Set background
set_background('bgs/bg5.jpg')

# Set title
st.title('Breast Cancer Classification')

# Text inputs for breast cancer prediction parameters
st.title('Breast Cancer Prediction Parameters Input')

# Create text inputs for each parameter
parameters = {
    'Mean Radius': st.text_input('Mean Radius'),
    'Mean Texture': st.text_input('Mean Texture'),
    'Mean Perimeter': st.text_input('Mean Perimeter'),
    'Mean Area': st.text_input('Mean Area'),
    'Mean Smoothness': st.text_input('Mean Smoothness'),
    'Mean Compactness': st.text_input('Mean Compactness'),
    'Mean Concavity': st.text_input('Mean Concavity'),
    'Mean Concave Points': st.text_input('Mean Concave Points'),
    'Mean Symmetry': st.text_input('Mean Symmetry'),
    'Mean Fractal Dimension': st.text_input('Mean Fractal Dimension'),
    'Radius Error': st.text_input('Radius Error'),
    'Texture Error': st.text_input('Texture Error'),
    'Perimeter Error': st.text_input('Perimeter Error'),
    'Area Error': st.text_input('Area Error'),
    'Smoothness Error': st.text_input('Smoothness Error'),
    'Compactness Error': st.text_input('Compactness Error'),
    'Concavity Error': st.text_input('Concavity Error'),
    'Concave Points Error': st.text_input('Concave Points Error'),
    'Symmetry Error': st.text_input('Symmetry Error'),
    'Fractal Dimension Error': st.text_input('Fractal Dimension Error'),
    'Worst Radius': st.text_input('Worst Radius'),
    'Worst Texture': st.text_input('Worst Texture'),
    'Worst Perimeter': st.text_input('Worst Perimeter'),
    'Worst Area': st.text_input('Worst Area'),
    'Worst Smoothness': st.text_input('Worst Smoothness'),
    'Worst Compactness': st.text_input('Worst Compactness'),
    'Worst Concavity': st.text_input('Worst Concavity'),
    'Worst Concave Points': st.text_input('Worst Concave Points'),
    'Worst Symmetry': st.text_input('Worst Symmetry'),
    'Worst Fractal Dimension': st.text_input('Worst Fractal Dimension')
}

# Add a button to submit the data
if st.button('Predict'):
    try:
        # Collect the entered data
        data = np.array([
            parameters['Mean Radius'], parameters['Mean Texture'], parameters['Mean Perimeter'], parameters['Mean Area'], parameters['Mean Smoothness'],
            parameters['Mean Compactness'], parameters['Mean Concavity'], parameters['Mean Concave Points'], parameters['Mean Symmetry'],
            parameters['Mean Fractal Dimension'], parameters['Radius Error'], parameters['Texture Error'], parameters['Perimeter Error'],
            parameters['Area Error'], parameters['Smoothness Error'], parameters['Compactness Error'], parameters['Concavity Error'],
            parameters['Concave Points Error'], parameters['Symmetry Error'], parameters['Fractal Dimension Error'], parameters['Worst Radius'],
            parameters['Worst Texture'], parameters['Worst Perimeter'], parameters['Worst Area'], parameters['Worst Smoothness'], parameters['Worst Compactness'],
            parameters['Worst Concavity'], parameters['Worst Concave Points'], parameters['Worst Symmetry'], parameters['Worst Fractal Dimension']
        ], dtype=float).reshape(1, -1)

        # Scale the input data
        data_scaled = scaler.transform(data)

        # Make predictions
        knn_prediction = knn.predict(data_scaled)
        knn_prediction_proba = knn.predict_proba(data_scaled)

        # Display KNN prediction
        st.subheader('KNN Prediction')
        knn_result = 'Malignant' if knn_prediction[0] == 1 else 'Benign'
        st.write(f'**{knn_result}**')
        st.write(f'**Probability**: {knn_prediction_proba[0][0]:.2f}')

        # If CNN model is loaded, make CNN prediction
        if model_loaded:
            # Load and preprocess the image for the CNN model
            uploaded_file = st.file_uploader("Upload a Mammogram Image", type=["jpg", "jpeg", "png", "pgm"])
            
            if uploaded_file is not None:
                # Load the image using PIL
                image = Image.open(uploaded_file).convert('RGB')  # Convert to RGB
                image_resized = image.resize((224, 224))  # Resize to the input size the CNN expects
                image_array = np.array(image_resized).reshape((1, 224, 224, 3)) / 255.0  # Normalize the image

                # Make a prediction using the CNN model
                cnn_prediction = cnn_model.predict(image_array)
                cnn_result = 'Malignant' if cnn_prediction[0][0] > 0.5 else 'Benign'
                cnn_confidence = cnn_prediction[0][0] if cnn_result == 'Malignant' else 1 - cnn_prediction[0][0]

                # Display CNN prediction
                st.subheader('CNN Prediction')
                st.write(f'**{cnn_result}**')
                st.write(f'**Confidence**: {cnn_confidence:.2f}')

                # Display the uploaded image
                st.image(image, caption='Uploaded Image', use_column_width=True)
            else:
                st.warning('Please upload an image to make CNN prediction.')

    except ValueError as e:
        st.error(f"ValueError: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred during prediction: {e}")

