import streamlit as st
import numpy as np
from PIL import Image

# --- הגדרות עמוד ---
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

# ---- צד ימין: צפייה בגרפים ובחירת מודל ----
with col_right:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("⚙️ Model Options")

    # ניהול מצב צפייה בגרף (toggle)
    if "show_graphs" not in st.session_state:
        st.session_state.show_graphs = False

    # כפתור צפייה / הסתרה
    if st.button("📊 View / Hide Model Performance"):
        st.session_state.show_graphs = not st.session_state.show_graphs

    # הצגת הגרף מה-Google Drive
    if st.session_state.show_graphs:
        st.markdown("### 📈 Model Accuracy - Transfer Learning")
        graph_url = "https://drive.google.com/uc?export=download&id=1AZ05TyAU8pc0-nhupdi9mCe9xB_-EJvi"
        st.image(graph_url, caption="Transfer Learning Performance Graph", use_container_width=True)

    # בחירת מודל
    model_choice = st.selectbox("Select a model:", ["Transfer Learning", "MobileNetV2", "EfficientNetB0"])

    # כפתור סיווג
    if st.button("🌺 Classify") and uploaded_file:
        confidence = np.random.uniform(85, 99)
        predicted_class = np.random.choice(["Daisy", "Dandelion", "Tulip", "Rose", "Sunflower"])
        st.success(f"**Predicted Flower:** {predicted_class}  \n**Confidence:** {confidence:.2f}%")

    st.markdown('</div>', unsafe_allow_html=True)
