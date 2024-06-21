import base64
import streamlit as st
from keras.models import load_model
from PIL import Image

from util import classify

# Function to set background
def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
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
model = load_model('oneclass.h5')
# load class names
with open('labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    class_name, conf_score = classify(image, model, class_names)

    # write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))



