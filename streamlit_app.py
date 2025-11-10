import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown
import os

# --- הורדת המודל מה־Drive אם לא קיים ---
file_id = "1vfNjI3VS-U5O30zc7MU6jTa7BwajcACU"
dl_url = f"https://drive.google.com/uc?id={file_id}"
model_file = "cnn_flowers_model.keras"

if not os.path.exists(model_file):
    gdown.download(dl_url, model_file, quiet=False)

# --- טעינת המודל ---
model = tf.keras.models.load_model(model_file)
class_names = ["Daisy", "Dandelion", "Tulip"]

# --- הגדרות עמוד ---
st.set_page_config(page_title="🌸 Flower Classifier", page_icon="🌺", layout="wide")

# --- עיצוב כללי ---
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom right, #f7f4ff, #e0f7fa);
    font-family: 'Segoe UI', sans-serif;
}
header {visibility: hidden;}

.title {
    text-align: center;
    font-size: 54px;
    color: #6C63FF;
    font-weight: 800;
    margin-top: 10px;
}
.subtitle {
    text-align: center;
    font-size: 20px;
    color: #333;
    margin-bottom: 25px;
}
.section {
    background-color: #ffffffcc;
    border-radius: 20px;
    padding: 35px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
}
.result-box {
    background-color: white;
    border-radius: 20px;
    padding: 35px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    text-align: center;
}
button[data-testid="baseButton-primary"] {
    background-color: #6C63FF !important;
    color: white !important;
    border-radius: 10px !important;
    font-size: 18px !important;
    padding: 10px 30px !important;
}
</style>
""", unsafe_allow_html=True)

# --- כותרת ראשית ---
st.markdown('<div class="title">🌷 Flower Classifier 🌷</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle"Let’s see if my model can guess your flower! Upload an image below 🌸 .</div>', unsafe_allow_html=True)

# --- עמודות זו לצד זו ---
col1, col2 = st.columns([1.2, 1])

# --- עמוד שמאלי: העלאה ---
with col1:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("📸 Upload Image")
    uploaded_file = st.file_uploader("Upload a flower image:", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- עמוד ימני: תוצאה ---
with col2:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("🔍 Prediction Results")
    if uploaded_file:
        img = image.resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        if st.button("Classify"):
            with st.spinner("Analyzing image... 🌸"):
                prediction = model.predict(img_array)
                predicted_class = class_names[np.argmax(prediction)]
                confidence = np.max(prediction) * 100

            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.write(f"**Flower Type:** {predicted_class}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("⬆️ Please upload an image first.")
    st.markdown('</div>', unsafe_allow_html=True)
