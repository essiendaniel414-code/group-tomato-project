import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

model = load_model("tomato_leaf_disease_model.keras")

st.title("Tomato Leaf Disease Detection")

st.write("Upload a tomato leaf image to classify it as Healthy or Early Blight.")

uploaded_file = st.file_uploader(
    "Choose a tomato leaf image",
    type=["jpg", "jpeg", "png", "jfif"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        st.success("Prediction: Tomato Healthy 🍅")
    else:
        st.error("Prediction: Tomato Early Blight 🍂")

        

