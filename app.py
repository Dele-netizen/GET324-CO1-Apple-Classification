import os
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

st.set_page_config(
    page_title="Fresh vs Rotten Apple Classifier",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "models/apple_classifier.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Fresh", "Rotten"]


with st.sidebar:
    st.title("🍎 Navigation")
    
    with st.container(border=True):
        st.subheader("📋 Project Info")
        st.write("**Course:** GET 324 Mini-Project")
        st.write("**Group:** CO1")
        st.write("**Dept:** Computer Engineering")
    
    with st.container(border=True):
        st.subheader("💡 How It Works")
        st.markdown(
            """
            1. **Upload** an image or select a sample.
            2. Image is resized to **224×224**.
            3. **MobileNetV3** model classifies state.
            4. Real-time confidence score is output.
            """
        )
    
    st.caption("🤖 Model: MobileNetV3Small (Transfer Learning)")

# 3. App Header
st.title("🍎 Fresh vs Rotten Apple Classifier")
st.write("Upload a fruit image or choose a pre-loaded sample below to analyze its freshness.")

# 4. Load Keras Model Safely
@st.cache_resource
def get_model():
    from tensorflow.keras.models import load_model
    return load_model(MODEL_PATH)

model = None
model_load_error = None

if not os.path.exists(MODEL_PATH):
    model_load_error = f"⚠️ Model file not found at `{MODEL_PATH}`. Please place the `.keras` file in the `models/` directory."
else:
    try:
        model = get_model()
    except Exception as e:
        model_load_error = f"⚠️ Failed to load model: {e}"

if model_load_error:
    st.error(model_load_error)
    st.stop()

# 5. Sample Selection Gallery
SAMPLES_DIR = "samples"
sample_choice = None

if os.path.isdir(SAMPLES_DIR):
    sample_files = [f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if sample_files:
        st.subheader("📁 Quick Test Samples")
        cols = st.columns(min(len(sample_files), 4))
        for idx, (col, fname) in enumerate(zip(cols, sample_files)):
            with col:
                img_path = os.path.join(SAMPLES_DIR, fname)
                st.image(img_path, use_container_width=True)
                btn_label = fname.split(".")[0].replace("_", " ").title()
                if st.button(f"Select {btn_label}", key=f"sample_{idx}", use_container_width=True):
                    sample_choice = img_path

st.divider()

# 6. Main Workspace (Upload & Results)
uploaded_file = st.file_uploader("Upload an Apple Image", type=["jpg", "jpeg", "png"])
image_source = uploaded_file if uploaded_file is not None else sample_choice

if image_source is not None:
    col_input, col_output = st.columns([1, 1], gap="medium")

    # Left Column: Input Image Card
    with col_input:
        with st.container(border=True):
            st.subheader("🖼️ Selected Image")
            try:
                img = Image.open(image_source).convert("RGB")
                st.image(img, use_container_width=True)
            except UnidentifiedImageError:
                st.error("Invalid image format. Please select a valid JPG or PNG.")
                st.stop()
            except Exception as e:
                st.error(f"Error opening image: {e}")
                st.stop()
            
            run_btn = st.button("🚀 Classify Apple", type="primary", use_container_width=True)

    # Right Column: Prediction Results Card
    with col_output:
        with st.container(border=True):
            st.subheader("📊 Analysis Results")
            
            if run_btn:
                from tensorflow.keras.preprocessing import image as keras_image

                # Preprocessing
                img_resized = img.resize(IMG_SIZE)
                img_array = keras_image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)

                with st.spinner("Analyzing image features..."):
                    try:
                        prob_rotten = float(model.predict(img_array, verbose=0)[0][0])
                        pred_idx = int(prob_rotten >= 0.5)
                        label = CLASS_NAMES[pred_idx]
                        confidence = prob_rotten if pred_idx == 1 else 1.0 - prob_rotten
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
                        st.stop()

                # Visual Feedback
                if label == "Fresh":
                    st.success(f"### 🎉 Result: {label} Apple", icon="✅")
                else:
                    st.error(f"### Result: {label} Apple", icon="⚠️")

                # Metrics display
                col_metric1, col_metric2 = st.columns(2)
                with col_metric1:
                    st.metric("Predicted State", label)
                with col_metric2:
                    st.metric("Confidence Score", f"{confidence * 100:.2f}%")

                st.write("**Confidence Level:**")
                st.progress(float(confidence))
                
            else:
                st.info("Click **Classify Apple** on the left to run inference.")

st.divider()
st.caption("GET 324 Lab Exercise 10 (Mini-Project) — Group CO1 • Computer Engineering Department")
