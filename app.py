import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image


st.set_page_config(
    page_title="Fresh vs Rotten Apple Classifier",
    page_icon="🍎",
    layout="centered",
)

IMAGE_SIZE = (224, 224)          # must match training IMAGE_HEIGHT / IMAGE_WIDTH
CLASS_NAMES = ["Fresh", "Rotten"]  # order must match train_dataset.class_names from training


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/apple_classifier.keras")
    return model

def predict(model, pil_image):
    """Preprocess the image and return the predicted label and class probabilities."""
    img = pil_image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)

    prob_rotten = float(model.predict(arr, verbose=0)[0][0])  # sigmoid output: P(class = "Rotten")
    prob_fresh = 1.0 - prob_rotten

    label = CLASS_NAMES[int(prob_rotten >= 0.5)]
    return label, prob_fresh * 100, prob_rotten * 100


st.title("🍎 Fresh vs Rotten Apple Classifier")
st.write(
    "Upload a photo of an apple and the model will predict whether it looks "
    "**fresh** or **rotten**."
)

model = load_model()
uploaded_file = st.file_uploader(
    "Upload an apple image", type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, width=300)

    label, fresh_pct, rotten_pct = predict(model, img)

    st.write(f"### Prediction: **{label}**")
    st.progress(int(fresh_pct), text=f"Fresh: {fresh_pct:.1f}%")
    st.progress(int(rotten_pct), text=f"Rotten: {rotten_pct:.1f}%")

    if label == "Fresh":
        st.success("This apple looks fresh! 🍏")
    else:
        st.warning("This apple looks rotten. 🍎⚠️")
else:
    st.info("Upload an image above to get a prediction.")
