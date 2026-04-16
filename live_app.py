import streamlit as st
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

st.set_page_config(page_title="Live Wire Inspector", layout="wide")
st.title("📹 Real-Time Wire Quality Scanner")
st.markdown("---")

# 1. REBUILD & LOAD MODEL (Weights-Only Method)
@st.cache_resource
def load_factory_model():
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
    m = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    weights_path = 'wire_weights.weights.h5'
    if os.path.exists(weights_path):
        m.build((None, 224, 224, 3)) 
        m.load_weights(weights_path)
        return m
    return None

model = load_factory_model()

# 2. CAMERA INITIALIZATION
if model is None:
    st.error("Weights file not found! Please check 'wire_weights.weights.h5'")
else:
    # Use Streamlit's image placeholder to update frames
    frame_placeholder = st.empty()
    stop_button = st.button("Stop Camera")
    
    # Access Laptop Camera
    cap = cv2.VideoCapture(0)

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to access camera.")
            break

        # --- STEP 3: OBJECT DETECTION (FINDING THE WIRE) ---
        # We use simple contours to find the wire and draw a bounding box
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 5000: # Ignore small noise, only track the wire
                x, y, w, h = cv2.boundingRect(cnt)
                
                # --- STEP 4: AI CLASSIFICATION ---
                # Crop the wire area for the AI model
                roi = frame[y:y+h, x:x+w]
                if roi.size > 0:
                    img_input = cv2.resize(roi, (224, 224))
                    img_input = img_input / 255.0
                    img_input = np.expand_dims(img_input, axis=0)
                    
                    prediction = model.predict(img_input, verbose=0)[0][0]
                    
                    # Determine Color and Label
                    if prediction < 0.5:
                        label = f"GOOD ({(1-prediction)*100:.1f}%)"
                        color = (0, 255, 0) # Green for Good
                    else:
                        label = f"NO GOOD ({prediction*100:.1f}%)"
                        color = (0, 0, 255) # Red for No Good
                    
                    # Draw the Box and Text on the original frame
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Convert BGR (OpenCV) to RGB (Streamlit/Web)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

    cap.release()
    st.write("Camera stopped.")