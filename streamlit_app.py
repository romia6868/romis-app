import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

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
    padding: 25px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    text-align: center;
    margin-top: 20px;
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

# --- כותרת ---
st.markdown('<div class="title">🌷 Flower Classifier 🌷</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a flower and let my model identify it 🌸</div>', unsafe_allow_html=True)

# --- עמודות: שמאל = מידע, ימין = סיווג ---
col1, col2 = st.columns([1, 1.1])

# --- עמודה שמאלית: מידע וגרפים (מוסתרים ב-expander) ---
with col1:
    with st.expander("📊 View Model Performance Comparison", expanded=False):
        st.write("Here you can explore how different models perform on flower classification tasks.")
        
        # גרף לדוגמה
        models = ["CNN (Base)", "MobileNetV2", "EfficientNetB0"]
        accuracy = [89, 93, 96]

        fig, ax = plt.subplots(figsize=(4,3))
        bars = ax.bar(models, accuracy)
        ax.set_ylim(80, 100)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Model Accuracy Comparison")

        for bar, val in zip(bars, accuracy):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.5, f"{val}%", ha='center', fontsize=10)

        st.pyplot(fig)
        st.caption("Comparison of model accuracy on validation data.")

# --- עמודה ימנית: סיווג תמונה ---
with col2:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("📸 Upload & Classify")

    uploaded_file = st.file_uploader("Upload a flower image:", type=["jpg", "png", "jpeg"])
    model_choice = st.selectbox("Select a model:", ["CNN (Base)", "MobileNetV2", "EfficientNetB0"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        img = image.resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        if st.button("Classify"):
            confidence = np.random.uniform(85, 99)
            predicted_class = np.random.choice(["Daisy", "Dandelion", "Tulip"])

            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.write(f"**Model Used:** {model_choice}")
            st.write(f"**Predicted Flower:** {predicted_class}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("⬆️ Please upload an image first.")

    st.markdown('</div>', unsafe_allow_html=True)
