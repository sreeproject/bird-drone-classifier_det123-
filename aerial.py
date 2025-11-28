import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch
import numpy as np

# ---------------------------------------------------------
# Load YOLOv8 Classification Model
# ---------------------------------------------------------
#CLASSIFICATION_MODEL_PATH = "best.pt"   # Your best model path
#model_cls = YOLO(CLASSIFICATION_MODEL_PATH)

# (Optional) Load YOLO Detection Model
DETECTION_MODEL_PATH = "best.pt"     # Use only if needed
model_det = YOLO(DETECTION_MODEL_PATH)

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.set_page_config(page_title="Bird vs Drone Classifier", layout="centered")
st.title("🛩️ YOLOv8 Aerial Image Classifier")
st.write("Upload an image to detect **Bird / Drone** with confidence score.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

mode = st.radio(
    "Select Mode",
    ["Classification (Bird / Drone)", "YOLO Detection (Bounding Boxes)"]
)

# ---------------------------------------------------------
# Process Image
# ---------------------------------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        st.write("🔍 Processing... please wait.")

        # -------------------------
        # Classification Mode
        # -------------------------
        if mode == "Classification (Bird / Drone)":
            #results = model_det.predict(image, imgsz=640)  
            results = model_det(image)
            probs = results[0].probs
            cls_id = probs.top1
            confidence = probs.top1conf.item()
            #cls_id = results[0].probs.top1
            #confidence = float(results[0].probs.top1conf)

            class_name = model_det.names[cls_id]

            st.subheader("📌 Prediction Result")
            st.success(f"**{class_name}**  (Confidence: {confidence:.2f})")

            # Show probabilities for all classes
            #st.subheader("🔢 Class Probabilities")
            #for i, prob in enumerate(confidence): #results[0].probs.data.tolist()):
                #st.write(f"{model_det.names[i]} : {prob:.4f}")

        # -------------------------
        # Detection Mode
        # -------------------------
        else:
            results = model_det.predict(image, save=False)

            st.subheader("📌 YOLO Detection Output")
            results[0].plot()  # This returns a numpy array

            output_img = Image.fromarray(results[0].plot())
            st.image(output_img, caption="YOLO Bounding Box Detection", use_container_width=True)

            #st.info("Bounding Boxes + Labels drawn using YOLOv8s detection model.")
