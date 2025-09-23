import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

# Load your trained model once globally
MODEL_PATH = "sugarcane_FIXED.keras"
model = tf.keras.models.load_model(MODEL_PATH)

class_labels = ["Healthy", "Mosaic", "RedRot", "Yellow", "Rust"]

treatment_dict = {
    "Healthy": "No treatment needed. Maintain good agricultural practices.",
    "Mosaic": "Use virus-free planting material and control insect vectors.",
    "RedRot": (
        "Remove and burn infected plants immediately. "
        "Improve soil drainage and use resistant varieties. "
        "Apply systemic fungicides."
    ),
    "Yellow": (
        "Yellow Leaf Disease may have reddish tinge due to sucrose. "
        "Monitor for severe symptoms and maintain plant nutrition."
    ),
    "Rust": (
        "Apply appropriate fungicides and remove infected leaves. "
        "Ensure good air circulation and avoid overhead irrigation."
    ),
}

CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to accept prediction


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))  # Change as per your model input size
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def predict(image_bytes):
    x = preprocess_image(image_bytes)
    preds = model.predict(x)[0]
    top_idx = np.argmax(preds)
    top_confidence = preds[top_idx]
    label = class_labels[top_idx]

    alt_preds = [
        (class_labels[i], conf) for i, conf in enumerate(preds) if i != top_idx and conf > 0.1
    ]

    return label, top_confidence, alt_preds


def main():
    st.set_page_config(
        page_title="Sugarcane Disease Detection",
        page_icon="🌾",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    # Custom CSS for modern style and smaller upload/camera inputs
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #a8c0ff, #3f2b96);
            color: white;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            padding: 2rem 1rem;
        }
        .block-container {
            padding: 2rem 3rem 3rem;
            max-width: 700px;
            margin: auto;
            background-color: rgba(0,0,0,0.4);
            border-radius: 20px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }
        h1 {
            font-weight: 700;
            font-size: 2.5rem;
            margin-bottom: 0;
            text-align: center;
            letter-spacing: 1.2px;
        }
        h4 {
            color: #d1d1d1;
            margin-top: 0.2rem;
            margin-bottom: 2rem;
            font-weight: 400;
            text-align: center;
        }
        .stFileUploader>div>div>input {
            max-width: 250px;
            background: #3f2b96;
            border-radius: 10px;
            color: white;
            padding: 0.75rem 1rem;
            cursor: pointer;
            font-size: 1rem;
            border: none;
        }
        div[data-testid="stCameraInput"] video {
            max-width: 250px;
            border-radius: 10px;
        }
        button[kind="primary"] {
            background-color: #2ecc71 !important;
            color: white !important;
            font-size: 1.1rem;
            padding: 0.5rem 2rem;
            border-radius: 12px;
            transition: background-color 0.3s ease;
        }
        button[kind="primary"]:hover {
            background-color: #27ae60 !important;
        }
        .stImage > img {
            border-radius: 20px;
            border: 4px solid #2ecc71;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        .result {
            background-color: rgba(46, 204, 113, 0.1);
            padding: 1rem 1.5rem;
            border-radius: 15px;
            color: white;
            border: 2px solid #2ecc71;
            margin-top: 1.5rem;
            font-size: 1.2rem;
        }
        .warning-msg {
            background-color: rgba(241, 196, 15, 0.2);
            padding: 0.8rem 1rem;
            border-radius: 10px;
            margin-top: 1rem;
            border: 1.5px solid #f1c40f;
            color: #f1c40f;
            font-weight: 600;
        }
        .alternative-preds {
            margin-top: 1rem;
            color: #ecf0f1;
            font-size: 0.95rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<h1>Sugarcane Disease Detector</h1>", unsafe_allow_html=True)
    st.markdown("<h4>Upload a clear image or capture from camera to detect leaf diseases.</h4>", unsafe_allow_html=True)

    # Side-by-side smaller upload and capture inputs
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded = st.file_uploader("Upload image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
    with col2:
        captured = st.camera_input("Capture from camera")

    image_bytes = None
    if uploaded is not None:
        image_bytes = uploaded.read()
    elif captured is not None:
        image_bytes = captured.getvalue()

    if image_bytes:
        st.image(image_bytes, caption="Selected Image", use_container_width=True)

        if st.button("Predict Disease"):
            with st.spinner("Analyzing image..."):
                label, confidence, alt_preds = predict(image_bytes)

            if confidence < CONFIDENCE_THRESHOLD:
                st.markdown(
                    '<div class="warning-msg">'
                    f"⚠️ Low confidence ({confidence:.2f}). Please upload a clearer image."
                    '</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<div class="result"><b>Prediction:</b> {label} ({confidence*100:.2f}%)<br><br>'
                f'<b>Treatment recommendation:</b><br>{treatment_dict.get(label, "No treatment info available.")}'
                "</div>",
                unsafe_allow_html=True,
            )

            if alt_preds:
                alt_html = '<div class="alternative-preds"><b>Other possible diseases:</b><ul>'
                for alt_label, alt_conf in alt_preds:
                    alt_html += f'<li>{alt_label}: {alt_conf*100:.2f}%</li>'
                alt_html += "</ul></div>"
                st.markdown(alt_html, unsafe_allow_html=True)

    else:
        st.info("Please upload an image or capture one using your device camera.")


if __name__ == "__main__":
    main()
