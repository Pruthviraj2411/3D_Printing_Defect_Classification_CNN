import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Set page config
st.set_page_config(page_title="Defect Detection", layout="centered")

st.title("3D Printing Defect Detection")

# Class labels from dataset
CLASS_NAMES = ['Cracking', 'Layer_shifting', 'Off_platform', 'Stringing', 'Warping']

from tensorflow.keras.applications.vgg16 import preprocess_input

# Load model
@st.cache_resource
def load_trained_model():
    model_paths = [
        "model.keras",
        "vgg16_model.keras",
        "model.h5",
        "defect_model.keras",
        "defect_model.h5"
    ]
    custom_objects = {"preprocess_input": preprocess_input}
    for path in model_paths:
        if os.path.exists(path):
            try:
                return tf.keras.models.load_model(path, custom_objects=custom_objects, safe_mode=False)
            except Exception:
                return tf.keras.models.load_model(path, custom_objects=custom_objects)
    return None

model = load_trained_model()

if model is None:
    st.warning("Model file not found. Please save your trained model as 'model.keras' in this directory.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if model is not None:
        # Preprocess image to match training input
        img = image.resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(np.max(predictions[0]) * 100)

        # Display result & accuracy/confidence
        st.subheader("Results")
        st.write(f"**Prediction:** {predicted_class}")
        st.write(f"**Accuracy / Confidence:** {confidence:.2f}%")
