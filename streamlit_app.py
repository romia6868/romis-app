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
    confidence = np.max(prediction) * 100   # 💡 להמיר לאחוזים
    st.success(f"המודל מזהה: **{predicted_class}** 🌼")
    st.write(f"✅ רמת ביטחון: {confidence:.2f}%")  # 💡 הוספת סימן אחוז


import streamlit as st

# --- הגדרות עמוד ---
st.set_page_config(page_title="🌸 Flower Classifier", page_icon="🌺", layout="wide")

# --- עיצוב כללי ---
st.markdown("""
<style>
/* רקע כללי */
[data-testid="stAppViewContainer"] {
    background-image: linear-gradient(to bottom right, #f7f4ff, #e0f7fa);
    background-attachment: fixed;
}

/* הסתרת הלוגו העליון */
header {visibility: hidden;}

/* כותרת ראשית */
.title {
    text-align: center;
    font-size: 60px;
    color: #6C63FF;
    font-weight: 800;
    margin-top: 20px;
}

/* תת־כותרת */
.subtitle {
    text-align: center;
    font-size: 22px;
    color: #333;
    margin-bottom: 40px;
}

/* עיצוב הקובץ המועלה */
.upload-section {
    text-align: center;
    border-radius: 20px;
    background-color: #ffffffcc;
    padding: 40px;
    margin: 0 auto;
    width: 60%;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}

/* עיצוב התוצאה */
.result-box {
    background-color: white;
    border-radius: 15px;
    padding: 25px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# --- כותרת ---
st.markdown('<div class="title">🌷 Flower Classifier 🌷</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a photo and let my models tell you which flower it is 🌺</div>', unsafe_allow_html=True)

st.markdown('<div class="upload-section">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a flower image:", type=["jpg", "png", "jpeg"])
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="result-box">', unsafe_allow_html=True)
st.subheader("Prediction Results 🌼")
st.write(f"**Flower Type:** {predicted_class}")
st.write(f"**Confidence:** {confidence:.2f}%")
st.markdown('</div>', unsafe_allow_html=True)
