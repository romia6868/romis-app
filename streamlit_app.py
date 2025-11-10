import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import gdown
import os

# הורדה אוטומטית של המודל מהמיקום בדרייב
file_id = "1vfNjI3VS-U5O30zc7MU6jTa7BwajcACU"  # מזהה מהקישור ששלחת
dl_url = f"https://drive.google.com/uc?id={file_id}"
model_file = "cnn_flowers_model.keras"
if not os.path.exists(model_file):
    gdown.download(dl_url, model_file, quiet=False)

# טעינת המודל
model = tf.keras.models.load_model(model_file)

class_names = ["Daisy", "Dandelion", "Tulip"]

st.set_page_config(page_title="🌷 Flower Classifier", page_icon="🌸", layout="centered")
st.title("🌸 Flower Classifier App")
st.write("העלו תמונה של פרח כדי לזהות את סוגו 🌼")

uploaded_file = st.file_uploader("בחרי תמונה (JPG או PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="תמונה שהועלתה", use_column_width=True)
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

if st.button("סווג את התמונה"):
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100   # אחוזים
    st.success(f"המודל מזהה: **{predicted_class}** 🌼")
    st.write(f"✅ רמת ביטחון: {confidence:.2f}%")

import streamlit as st
import numpy as np
from PIL import Image

# --- הגדרות עמוד ---
st.set_page_config(page_title="🌸 Flower Classifier", page_icon="🌺", layout="centered")

# --- CSS לעיצוב כולל ---
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom right, #f8f6ff, #e8faff);
    background-attachment: fixed;
    font-family: 'Segoe UI', sans-serif;
}

/* הסתרת header */
header {visibility: hidden;}

/* כותרת */
.title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    color: #6C63FF;
    margin-top: 10px;
}

/* תת-כותרת */
.subtitle {
    text-align: center;
    font-size: 20px;
    color: #333;
    margin-bottom: 35px;
}

/* אזור העלאה */
.upload-box {
    background: white;
    border-radius: 25px;
    padding: 30px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    width: 80%;
    margin: 0 auto;
    text-align: center;
}

/* תיבת תוצאה */
.result-box {
    background: #ffffffee;
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    text-align: center;
    margin-top: 30px;
}

/* עיצוב כפתור */
button[data-testid="baseButton-primary"] {
    background-color: #6C63FF !important;
    color: white !important;
    border-radius: 10px !important;
    font-size: 18px !important;
    padding: 10px 30px !important;
}
</style>
""", unsafe_allow_html=True)

# --- כותרת עליונה ---
st.markdown('<div class="title">🌷 Flower Classifier 🌷</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a flower photo and get instant AI classification!</div>', unsafe_allow_html=True)

# --- תיבת העלאה ממורכזת ---
st.markdown('<div class="upload-box">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload image here", type=["jpg", "png", "jpeg"])
st.markdown('</div>', unsafe_allow_html=True)

# --- תוצאה ---
if uploaded_file:
    img = Image.open(uploaded_file)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # אזור כפתור ותחזית
    if st.button("🔍 Classify"):
        # דוגמה בלבד (תחליפי במודל שלך):
        predicted_class = "Rose"
        confidence = 97.8

        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.subheader("Prediction Results 🌼")
        st.write(f"**Flower Type:** {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("⬆️ Upload an image above to get started.")
