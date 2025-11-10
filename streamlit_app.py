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
        confidence = np.max(prediction)
        st.success(f"המודל מזהה: **{predicted_class}** 🌼")
        st.write(f"✅ רמת ביטחון: {confidence:.2f}")
