
import streamlit as st
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import gdown
import tensorflow as tf
import os

st.set_page_config(page_title="🌺 Flower Classifier", page_icon="🌺", layout="wide")

# ===================================================================
#                  הגדרות הורדת מודלים מהערך
# ===================================================================

DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1qb_T3GmxYJxM1nWc7fu-JllEslCVe-Gv?usp=drive_link"
LOCAL_MODEL_DIR = "models_from_drive"

MODEL_FILES = {
    "CNN": "cnn_flowers_model.keras",
    "Fine tuning": "fine_tuned_resnet50_final.h5",
    "Fully connected": "flowers_model.h5"
}

if not os.path.exists(LOCAL_MODEL_DIR):
    os.makedirs(LOCAL_MODEL_DIR)

# יצירת session state למודל טעון
if "loaded_model" not in st.session_state:
    st.session_state.loaded_model = None
if "loaded_model_name" not in st.session_state:
    st.session_state.loaded_model_name = None


# פונקציה שמורידה את כל התיקייה בדרייב
def download_all_models():
    try:
        gdown.download_folder(DRIVE_FOLDER_URL, output=LOCAL_MODEL_DIR, quiet=False)
        return True
    except Exception as e:
        st.error(f"Error downloading from Drive: {e}")
        return False


# טוען מודל מסוים לפי בחירה
def load_selected_model(model_name):
    file_name = MODEL_FILES[model_name]
    path = os.path.join(LOCAL_MODEL_DIR, file_name)

    if not os.path.exists(path):
        st.warning("Model file missing locally. Downloading from Drive...")
        if not download_all_models():
            return None

    try:
        model = tf.keras.models.load_model(path)
        st.session_state.loaded_model = model
        st.session_state.loaded_model_name = model_name
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


# ===================================================================
#                            עיצוב
# ===================================================================
st.markdown("""
<style>

/* טעינת הפונט Comfortaa */
@import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@300;400;600;700&display=swap');

/* ----- Comfortaa לכל האתר ----- */
div, p, span, button, label, input, textarea, select,
h1, h2, h3, h4, h5, h6 {
    font-family: 'Comfortaa', sans-serif !important;
}

/* ----- רקע כללי ----- */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(
        to bottom right,
        #ffe5ec,
        #ffc2d1
    );
}

/* ----- כותרת ----- */
.title {
    text-align: center;
    font-size: 50px;
    color: #fb6f92;
    font-weight: 700;
    margin-top: 5px;
}

/* ----- כותרת משנה ----- */
.subtitle {
    text-align: center;
    font-size: 20px;
    color: #6b6b6b;
    margin-bottom: 40px;
}

/* ----- קופסאות (section) ----- */
.section {
    background-color: #ffe5ec;
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.08);
}

/* ----- כפתורים ----- */
.stButton > button {
    background-color: #ff8fab !important;
    color: white !important;
    border-radius: 10px !important;
    font-size: 18px !important;
    padding: 8px 28px !important;
    border: none !important;
    transition: 0.2s ease-in-out;
}

/* אפקט רחף */
.stButton > button:hover {
    background-color: #fb6f92 !important;
    transform: scale(1.03);
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🎀 Flower Classifier 🎀</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a flower and let my model identify it!!☁️ </div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1.2, 1])

# ===================================================================
#                          העלאת תמונה
# ===================================================================

with col_left:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("📸 Upload Image please")

    uploaded_file = st.file_uploader("Choose a flower image:", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Flower", use_container_width=True)
    else:
        st.info("⬆️ Please upload an image to get started.")

    st.markdown('</div>', unsafe_allow_html=True)

# ===================================================================
#                          בחירת מודל + טעינה + חיזוי
# ===================================================================

with col_right:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("⚙️ Model Options")

    # הצגת גרפים
    if "show_graphs" not in st.session_state:
        st.session_state.show_graphs = False

    if st.button("📊 View / Hide Model Performance"):
        st.session_state.show_graphs = not st.session_state.show_graphs

    if st.session_state.show_graphs:
        st.markdown("### 📈 Model Accuracy - Transfer Learning")
        image_url = "https://drive.google.com/uc?export=view&id=1AZ05TyAU8pc0-nhupdi9mCe9xB_-EJvi"
        try:
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            st.image(img, caption="Transfer Learning Performance Graph", use_container_width=True)
        except:
            st.error("⚠️ Could not load graph.")

    # בחירת מודל
    model_choice = st.selectbox("Select a model:", list(MODEL_FILES.keys()))

    # כפתור טעינת המודל
    if st.button("🔌 Load Model"):
        st.write("🔄 Loading selected model...")
        m = load_selected_model(model_choice)
        if m is not None:
            st.success(f"✔️ Model '{model_choice}' loaded successfully")

    # הצגת סטטוס טעינה
    if st.session_state.loaded_model is not None:
        st.info(f"Loaded model: {st.session_state.loaded_model_name}")

    # כפתור סיווג
    if st.button("🌺 Classify") and uploaded_file:
        if st.session_state.loaded_model is None:
            st.error("⚠️ Please load a model before classifying.")
        else:
            st.write("🔄 Analyzing image...")

            model = st.session_state.loaded_model

            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)[0]
            class_names = ["Daisy", "Dandelion", "Tulip"]
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

             st.success(
                f"Looks like a {predicted_class} to me.\n"
                f"I'm {confidence:.2f}% sure — that feels illegal to be that good🧿."
                )
 

    st.markdown('</div>', unsafe_allow_html=True)

