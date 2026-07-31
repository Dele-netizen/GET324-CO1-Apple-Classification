import os
import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError

# Page setup
st.set_page_config(
    page_title="Fresh vs Rotten Apple Classifier",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Pure Python + CSS Injection)
st.markdown("""
    <style>
    /* Main Background & Fonts */
    .main {
        background-color: #0E1117;
    }
    
    /* Card Containers */
    .css-card {
        background-color: #1E222A;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #2D3139;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Custom Headers */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #9CA3AF;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    /* Status Badges */
    .badge-fresh {
        background-color: rgba(34, 197, 94, 0.15);
        color: #4ADE80;
        border: 1px solid #22C55E;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 12px;
    }
    .badge-rotten {
        background-color: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid #EF4444;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 12px;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161920;
        border-right: 1px solid #2D3139;
    }
    
    /* Primary Action Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        transition: all 0.2s ease;
    }
    </style>
""", unsafe_allow_html=True)

MODEL_PATH = "models/apple_classifier.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Fresh", "Rotten"]

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 📌 About Project")
    st.info("**GET 324 Mini-Project** | Group CO1\n\nComputer Engineering Department")
    
    st.markdown("---")
    st.markdown("### ⚙️ How It Works")
    st.markdown(
        """
        1. **Input:** Upload an image or select a pre-loaded sample.
        2. **Preprocessing:** Image is scaled to $224 \\times 224$ px.
        3. **Inference:** MobileNetV3 CNN evaluates features.
        4. **Output:** Class label & confidence percentage returned.
        """
    )
    st.markdown("---")
    st.caption("🤖 **Architecture:** MobileNetV3Small (Transfer Learning)")

# --- MAIN CONTENT ---
st.markdown('<div class="main-title">🍎 Fresh vs Rotten Apple Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload an image of an apple to run real-time quality classification.</div>', unsafe_allow_html=True)

# Cache model loading
@st.cache_resource
def get_model():
    from tensorflow.keras.models import load_model
    return load_model(MODEL_PATH)

model = None
model_load_error = None

if not os.path.exists(MODEL_PATH):
    model_load_error = f"Model file not found at `{MODEL_PATH}`. Please place the model file in the `models/` directory."
else:
    try:
        model = get_model()
    except Exception as e:
        model_load_error = f"Failed to load the model: {e}"

if model_load_error:
    st.error(model_load_error)
    st.stop()

# Interactive Sample Gallery
SAMPLES_DIR = "samples"
sample_choice = None

if os.path.isdir(SAMPLES_DIR):
    sample_files = [f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if sample_files:
        st.markdown("##### 📁 Or test with a sample image:")
        cols = st.columns(min(len(sample_files), 4))
        for idx, (col, fname) in enumerate(zip(cols, sample_files)):
            with col:
                img_path = os.path.join(SAMPLES_DIR, fname)
                st.image(img_path, use_container_width=True)
                clean_name = fname.split(".")[0].replace("_", " ").title()
                if st.button(f"Use {clean_name}", key=f"sample_{idx}"):
                    sample_choice = img_path

# File Upload Section
uploaded_file = st.file_uploader("Choose an apple image...", type=["jpg", "jpeg", "png"])
image_source = uploaded_file if uploaded_file is not None else sample_choice

# Layout Split: Display Input Image + Results side by side
if image_source is not None:
    st.markdown("---")
    col_img, col_res = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown("#### Selected Image")
        try:
            img = Image.open(image_source).convert("RGB")
            st.image(img, use_container_width=True)
        except UnidentifiedImageError:
            st.error("Invalid image format. Upload a valid JPG or PNG.")
            st.stop()
        except Exception as e:
            st.error(f"Error opening image: {e}")
            st.stop()
        
        classify_btn = st.button("🔍 Run Classification", type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_res:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown("#### Analysis & Result")
        
        if classify_btn:
            from tensorflow.keras.preprocessing import image as keras_image

            # Preprocess image
            img_resized = img.resize(IMG_SIZE)
            img_array = keras_image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)

            with st.spinner("Analyzing visual features..."):
                try:
                    prob_rotten = float(model.predict(img_array, verbose=0)[0][0])
                    pred_idx = int(prob_rotten >= 0.5)
                    label = CLASS_NAMES[pred_idx]
                    confidence = prob_rotten if pred_idx == 1 else 1.0 - prob_rotten
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.stop()

            # Result Badge
            if label == "Fresh":
                st.markdown('<div class="badge-fresh">✅ FRESH APPLE</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="badge-rotten">⚠️ ROTTEN APPLE</div>', unsafe_allow_html=True)

            # Confidence Score Representation
            st.metric(label="Model Confidence Score", value=f"{confidence * 100:.2f}%")
            st.progress(float(confidence))

            st.markdown(
                f"""
                **Summary:**  
                The vision model predicts with **{confidence * 100:.2f}%** certainty that this sample belongs to the **{label.lower()}** category.
                """
            )
        else:
            st.info("Click **Run Classification** on the left to analyze the selected image.")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("GET 324 Lab Exercise 10 (Mini-Project) — Group CO1 • Computer Engineering Department")
