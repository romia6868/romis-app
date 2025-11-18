import streamlit as st
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import gdown
import tensorflow as tf
import os

st.set_page_config(page_title="🌺 Flower Classifier", page_icon="🌺", layout="wide")

#טעינת המודלים
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1qb_T3GmxYJxM1nWc7fu-JllEslCVe-Gv?usp=drive_link"
LOCAL_MODEL_DIR = "models_from_drive"

MODEL_FILES = {
    "CNN (Base)": "cnn_flowers_model.keras",
    "Fine tuning": "fine_tuned_resnet50_final.h5",
    "Fully Connected": "flowers_model.h5"
}

if not os.path.exists(LOCAL_MODEL_DIR):
    os.makedirs(LOCAL_MODEL_DIR)

missing = [name for name in MODEL_FILES.values()
           if not os.path.exists(os.path.join(LOCAL_MODEL_DIR, name))]

if missing:
    st.write("⏳ מוריד את המודלים מה־Google Drive ...")
    try:
        gdown.download_folder(DRIVE_FOLDER_URL, output=LOCAL_MODEL_DIR, quiet=False)
        st.success("✔️ כל המודלים הורדו בהצלחה")
    except Exception as e:
        st.error(f"❌ שגיאה בהורדת המודלים: {e}")
        st.stop()

# טעינת המודלים לזיכרון
loaded_models = {}
for model_name, file_name in MODEL_FILES.items():
    path = os.path.join(LOCAL_MODEL_DIR, file_name)
    try:
        loaded_models[model_name] = tf.keras.models.load_model(path)
    except Exception as e:
        st.error(f"❌ שגיאה בטעינת {file_name}: {e}")
        st.stop()
#                             עיצוב

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

st.markdown('<div class="title">🌷 Flower Classifier 🌷</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a flower and let the model identify it 🌸</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1.2, 1])
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

#בחירת מודל
with col_right:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("⚙️ Model Options")

    # Graph toggles
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

    # Model selector
    model_choice = st.selectbox("Select a model:", list(MODEL_FILES.keys()))

    # Real prediction
    if st.button("🌺 Classify") and uploaded_file:
        st.write("🔄 Analyzing image...")

        model = loaded_models[model_choice]

        img = image.resize((224, 224))  # התאם לגודל שהמודלים שלך דורשים
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0]
        class_names = ["Daisy", "Dandelion", "Tulip", "Rose", "Sunflower"]
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.success(f"**🌼 Predicted Flower:** {predicted_class}\n**Confidence:** {confidence:.2f}%")

    st.markdown('</div>', unsafe_allow_html=True)
