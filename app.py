import base64
import streamlit as st
from keras.models import load_model
from PIL import Image
from util import classify  # Assuming classify function is defined in util.py

# Function to set background
def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

# Set background
set_background('bgs/bg5.jpg')

# Set title
st.title('Breast Cancer classification')

# Set header
st.header('Please upload a Breast Mammography image')

# Upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png', 'pgm'])

# Load classifier
try:
    model = load_model('oneclasso.keras')
    with open('labels.txt', 'r') as f:
        class_names = [a.strip().split(' ')[1] for a in f.readlines()]
except FileNotFoundError:
    st.error("Model file 'oneclass.h5' or labels file 'labels.txt' not found. Please check your files and try again.")
    st.stop()

# Display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # Classify image
    class_name, conf_score = classify(image, model, class_names)

    # Write classification
    st.write("## {}".format(class_name))
    st.write("### Score: {}%".format(int(conf_score * 1000) / 10))



