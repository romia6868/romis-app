import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="🌸 Flower Classifier", page_icon="🌺", layout="wide")

# --- עיצוב כללי ---
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
button[data-testid="baseButton-primary"] {
    background-color: #6C63FF !important;
    color: white !important;
    border-radius: 10px !important;
    font-size: 18px !important;
    padding: 8px 28px !important;
}
</style>
""", unsafe_allow_html=True)

# --- כותרת ---
st.markdown('<div class="title">🌷 Flower Classifier 🌷</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a flower and let the model identify it 🌸</div>', unsafe_allow_html=True)

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

# ---- צד ימין: בחירת מודל + סיווג + גרפים ----
with col_right:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("⚙️ Model Options")

    model_choice = st.selectbox("Select a model:", ["CNN (Base)", "MobileNetV2", "EfficientNetB0"])

    classify_btn = st.button("🌺 Classify Flower")

    if classify_btn and uploaded_file:
        # כאן תכניסי את הקריאה למודל שלך
        confidence = np.random.uniform(85, 99)
        predicted_class = np.random.choice(["Daisy", "Dandelion", "Tulip", "Rose", "Sunflower"])
        st.success(f"**Predicted Flower:** {predicted_class}  \n**Confidence:** {confidence:.2f}%")

    # כפתור לצפייה בגרפים
    show_graphs = st.checkbox("📊 View Model Performance")

    if show_graphs:
        models = ["CNN (Base)", "MobileNetV2", "EfficientNetB0"]
        accuracy = [89, 93, 96]

        fig, ax = plt.subplots(figsize=(4,3))
        bars = ax.bar(models, accuracy, color=["#9b8fff", "#6C63FF", "#4dd0e1"])
        ax.set_ylim(80, 100)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Model Accuracy Comparison")

        for bar, val in zip(bars, accuracy):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.5, f"{val}%", ha='center', fontsize=10)

        st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)
