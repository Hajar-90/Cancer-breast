import streamlit as st
import tensorflow as tf

@st.cache(allow_output_mutation=True)
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

model_path = 'oneone.keras'
model = load_model(model_path)

