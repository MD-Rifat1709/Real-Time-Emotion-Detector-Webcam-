# app.py
import streamlit as st
import cv2
import numpy as np
from keras.models import load_model

st.set_page_config(page_title="Real-Time Emotion Detector")
st.title("🎭 Real-Time Emotion Detector")

# Model Options
model_option = st.selectbox("Choose model to use:", ["Pretrained Model", "Custom Trained Model"])
model_path = "model/emotion_model.h5" if model_option == "Pretrained Model" else "model/custom_emotion_model.h5"

# Load model & classifier
model = load_model(model_path)
face_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

img_data = st.camera_input("📸 Take a picture")

if img_data:
    img = cv2.imdecode(np.frombuffer(img_data.getvalue(), np.uint8), 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48)) / 255.0
        roi = np.expand_dims(roi, axis=0)[..., np.newaxis]
        pred = model.predict(roi)[0]
        emotion = labels[np.argmax(pred)]
        st.success(f"Detected Emotion: **{emotion}**")

        # Visualization
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(img, emotion, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    st.image(img, channels="BGR", caption="Processed Image")
