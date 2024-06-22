import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model

# Load pre-trained ResNet50 model for demonstration
model = tf.keras.applications.ResNet50(weights='imagenet')
model = Model(inputs=model.input, outputs=(model.layers[-3].output, model.output))

# Function to generate Grad-CAM heatmap
def generate_gradcam(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    with tf.GradientTape() as tape:
        last_conv_layer, preds = model(x)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    grads = tape.gradient(top_class_channel, last_conv_layer)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer[0]
    heatmap = tf.reduce_mean(tf.multiply(last_conv_layer_output, pooled_grads), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap.numpy()

# Streamlit app
st.title('CNN Prediction Visualization with Grad-CAM')
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Generate Grad-CAM heatmap
    image = Image.open(uploaded_file)
    heatmap = generate_gradcam(uploaded_file)

    # Plot Grad-CAM heatmap
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(image)
    axs[0].axis('off')
    axs[0].set_title('Original Image')

    axs[1].imshow(heatmap, cmap='jet', alpha=0.6)
    axs[1].imshow(image, alpha=0.4)
    axs[1].axis('off')
    axs[1].set_title('Grad-CAM Heatmap')

    st.pyplot(fig)


