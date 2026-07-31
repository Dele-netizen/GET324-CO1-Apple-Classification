import os
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

# Page setup
st.set_page_config(
    page_title="Fresh vs Rotten Apple Classifier",
    page_icon="🍎",
    layout="centered"
)

MODEL_PATH = "models/apple_classifier.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Fresh", "Rotten"]

# Load model
@st.cache_resource
def load_classification_model():
    from tensorflow.keras.models import load_model
    return load_model(MODEL_PATH)

# Helper function to calibrate/scale overconfident probabilities
def calibrate_confidence(raw_prob, temperature=3.5):
    """
    Applies temperature scaling to soften overconfident CNN predictions.
    A higher temperature value brings confidence closer to realistic ranges (60%-90%).
    """
    # Convert raw probability back to logit
    eps = 1e-7
    raw_prob = np.clip(raw_prob, eps, 1 - eps)
    logit = np.log(raw_prob / (1 - raw_prob))
    
    # Scale logit with temperature
    scaled_logit = logit / temperature
    
    # Convert back to calibrated probability
    calibrated_prob = 1 / (1 + np.exp(-scaled_logit))
    return calibrated_prob

# Sidebar - Project Metadata
with st.sidebar:
    st.header("Project Info")
    st.write("**GET 324 Mini-Project**")
    st.write("Group CO1 | Computer Engineering")
    
    st.markdown("---")
    st.header("How it works")
    st.write(
        "Upload an image of an apple. The MobileNetV3 model will "
        "preprocess the image and classify it as Fresh or Rotten "
        "along with a confidence score."
    )
    st.caption("Model: MobileNetV3Small")

# Main Header
st.title("Fresh vs Rotten Apple Classifier")
st.write("Upload an image below to test the classifier.")

# Check for model existence
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at `{MODEL_PATH}`. Please check the path and try again.")
    st.stop()

try:
    model = load_classification_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Quick test samples section
SAMPLES_DIR = "samples"
selected_sample = None

if os.path.isdir(SAMPLES_DIR):
    sample_files = [f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if sample_files:
        st.write("**Or select a sample image:**")
        cols = st.columns(len(sample_files))
        for col, fname in zip(cols, sample_files):
            with col:
                img_path = os.path.join(SAMPLES_DIR, fname)
                st.image(img_path, use_container_width=True)
                if st.button(fname.split(".")[0].replace("_", " ").title(), key=f"btn_{fname}"):
                    selected_sample = img_path

# File uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
image_source = uploaded_file if uploaded_file is not None else selected_sample

# Prediction logic
if image_source is not None:
    st.markdown("---")
    
    try:
        img = Image.open(image_source).convert("RGB")
    except UnidentifiedImageError:
        st.error("Invalid image file. Please upload a valid JPG or PNG.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading image: {e}")
        st.stop()

    # Display uploaded image
    st.image(img, caption="Target Image", width=350)

    if st.button("Classify Image", type="primary"):
        from tensorflow.keras.preprocessing import image as keras_image

        # Preprocess image
        img_resized = img.resize(IMG_SIZE)
        img_array = keras_image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("Classifying..."):
            try:
                raw_prob_rotten = float(model.predict(img_array, verbose=0)[0][0])
                
                # Apply temperature scaling to prevent 99.9% overconfidence
                calibrated_prob = calibrate_confidence(raw_prob_rotten, temperature=3.5)
                
                pred_idx = int(calibrated_prob >= 0.5)
                label = CLASS_NAMES[pred_idx]
                confidence = calibrated_prob if pred_idx == 1 else 1.0 - calibrated_prob
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.stop()

        # Simple result presentation
        st.subheader("Result")
        if label == "Fresh":
            st.success(f"**Classification:** {label} Apple (Confidence: {confidence * 100:.1f}%)")
        else:
            st.error(f"**Classification:** {label} Apple (Confidence: {confidence * 100:.1f}%)")

st.markdown("---")
st.caption("GET 324 Lab Exercise 10 — Group CO1")
