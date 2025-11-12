import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# --- הגדרות כלליות ---
st.set_page_config(page_title="🌸 Flower Classifier", page_icon="🌺", layout="wide")

# --- עיצוב עמוד ---
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom right, #f8f9ff, #e0f7fa);
    font-family: 'Segoe UI', sans-serif;
}
.title {
    text-align: center;
    font-size: 50px;
    color: #6C63FF;
    font-weight: 800;
    margin-top: 5px;
}
.subtitle {
    text-align: center;
    font-size: 20px;
    color: #333;
    margin-bottom: 40px;
}
.section {
    background-color: white;
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.1);
}
.stButton > button {
    background-color: #6C63FF !important;
    color: white !important;
    border-radius: 10px !important;
    font-size: 18px !important;
    padding: 8px 28px !important;
    border: none !important;
}
</style>
""", unsafe_allow_html=True)

# --- כותרת ---
st.markdown('<div class="title">🌷 Flower Classifier 🌷</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a flower and let my model identify it 🌸</div>', unsafe_allow_html=True)

# --- תיקיית המודלים ---
MODELS_DIR = "/content/drive/MyDrive/flower_project"

# --- רשימת מודלים (תשני את הנתיבים שלך כאן) ---
model_files = {
    "Transfer Model": f"{MODELS_DIR}/transfer_resnet50_dense256.h5",
    "Fine-tuned Model": f"{MODELS_DIR}/fine_tuned_resnet50_final.h5",
    "Base CNN Model": f"{MODELS_DIR}/baseline_model.h5"
}

# --- שתי עמודות ---
col_left, col_right = st.columns([1.2, 1])

# ---- צד שמאל: העלאת תמונה ----
with col_left:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("📸 Upload Image")

    uploaded_file = st.file_uploader("Choose a flower image:", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Flower", use_container_width=True)
    else:
        st.info("⬆️ Please upload an image to get started.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---- צד ימין: בחירת מודל וניבוי ----
with col_right:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("⚙️ Model Options")

    # בחירת מודל אמיתי
    selected_model_name = st.selectbox("בחר מודל לאימון:", list(model_files.keys()))
    selected_model_path = model_files[selected_model_name]

    # טעינת המודל
    if st.button("🔄 Load Model"):
        try:
            st.session_state.model = tf.keras.models.load_model(selected_model_path)
            st.success(f"✅ Model '{selected_model_name}' loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")

    # ניבוי
    if st.button("🌺 Classify") and uploaded_file:
        if "model" not in st.session_state:
            st.warning("Please load a model first!")
        else:
            model = st.session_state.model

            # הכנת תמונה
            img = image.resize((224, 224))
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

            # ניבוי
            preds = model.predict(img_array)
            class_names = ["Daisy", "Dandelion", "Rose", "Sunflower", "Tulip"]  # תשני לפי המידע שלך
            predicted_class = class_names[np.argmax(preds)]
            confidence = np.max(preds) * 100

            st.success(f"**Predicted Flower:** {predicted_class}  \n**Confidence:** {confidence:.2f}%")

    st.markdown('</div>', unsafe_allow_html=True)
