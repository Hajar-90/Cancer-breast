import os
import streamlit as st
import tensorflow as tf

# Define absolute path to the .keras file
model_path = '/mount/src/Cancer-breast/oneone.keras'

# Load CNN model with error handling
model_loaded = False
try:
    cnn_model = tf.keras.models.load_model(model_path)
    model_loaded = True
except FileNotFoundError:
    st.error(f"CNN model file '{model_path}' not found. Please ensure the file exists.")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")

