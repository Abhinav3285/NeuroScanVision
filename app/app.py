import streamlit as st
import sys
import os
import io
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
from ultralytics import YOLO

# -------------------------------------------------
# FIX PYTHON PATH (VERY IMPORTANT)
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# -------------------------------------------------
# OPTIONAL: IMPORT SEGMENTATION SAFELY
# -------------------------------------------------
SEGMENTATION_AVAILABLE = False
UNET_PATH = os.path.join(PROJECT_ROOT, "model", "segmentation", "unet_best.pth")

if os.path.exists(UNET_PATH):
    try:
        from model.segmentation.predict_unet import predict_mask
        SEGMENTATION_AVAILABLE = True
    except Exception as e:
        SEGMENTATION_AVAILABLE = False

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AI Brain Tumor Detection System",
    layout="wide",
    page_icon="🧠"
)

# -------------------------------------------------
# LOAD YOLO MODEL
# -------------------------------------------------
@st.cache_resource
def load_yolo():
    return YOLO(
        os.path.join(
            PROJECT_ROOT,
            "results",
            "brain_tumor_run",
            "weights",
            "best.pt"
        )
    )

model = load_yolo()

# -------------------------------------------------
# YOLO GRAD-CAM STYLE HEATMAP (SAFE)
# -------------------------------------------------
def generate_yolo_cam(image_np, results):
    heatmap = np.zeros(image_np.shape[:2], dtype=np.float32)

    if results[0].boxes is None:
        return image_np

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        heatmap[y1:y2, x1:x2] += conf

    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)
    return overlay

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("🧠 AI Brain Tumor Detection System")
st.caption("⚠️ AI-assisted prediction. Not for medical diagnosis.")

with st.sidebar:
    st.header("🛠️ Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload MRI scan(s)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# -------------------------------------------------
# PROCESS IMAGES
# -------------------------------------------------
if uploaded_files:
    for uploaded_file in uploaded_files:

        st.divider()
        st.subheader(f"📄 File: {uploaded_file.name}")

        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)

        col1, col2 = st.columns(2)

        # ORIGINAL
        with col1:
            st.subheader("🖼️ Original MRI")
            st.image(image, use_container_width=True)
            st.write(f"Resolution: {image.size[0]} × {image.size[1]}")
            st.write("Slice Type: Axial (assumed)")

        # YOLO DETECTION
        with col2:
            st.subheader("🎯 YOLO Detection")
            results = model.predict(img_np, conf=conf_threshold)
            detected_img = Image.fromarray(results[0].plot())
            st.image(detected_img, use_container_width=True)

        # -------------------------------------------------
        # EXPLAINABLE AI
        # -------------------------------------------------
        st.subheader("🔥 Explainable AI (Grad-CAM)")
        cam_overlay = generate_yolo_cam(img_np, results)
        st.image(
            cam_overlay,
            caption="Highlighted regions influencing prediction",
            use_container_width=True
        )

        # -------------------------------------------------
        # SEGMENTATION (ONLY IF AVAILABLE)
        # -------------------------------------------------
        st.subheader("🧩 Tumor Segmentation (U-Net)")

        if SEGMENTATION_AVAILABLE:
            mask = predict_mask(img_np)

            overlay = img_np.copy()
            overlay[mask > 0] = [255, 0, 0]
            overlay = cv2.addWeighted(img_np, 0.7, overlay, 0.3, 0)

            st.image(
                overlay,
                caption="Precise Tumor Boundary (Segmentation)",
                use_container_width=True
            )
        else:
            st.warning(
                "U-Net segmentation model not found.\n\n"
                "Train U-Net to enable tumor shape visualization."
            )

        # -------------------------------------------------
        # TUMOR DETAILS
        # -------------------------------------------------
        st.subheader("🧪 Tumor Details")

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            st.success(f"Detected {len(boxes)} tumor(s)")

            for i, box in enumerate(boxes):
                label = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0]
                area = int((x2 - x1) * (y2 - y1))

                if area < 5000:
                    stage = "Stage I (Low)"
                elif area < 15000:
                    stage = "Stage II (Moderate)"
                else:
                    stage = "Stage III (High)"

                with st.container(border=True):
                    st.markdown(f"### 🧠 Tumor {i+1}")
                    st.write(f"Type: **{label.upper()}**")
                    st.write(f"Size: **{area:,} px²**")
                    st.write(f"Stage: **{stage}**")
                    st.progress(conf)

        else:
            st.warning("No tumors detected.")

        st.info(
            "⚠️ Detection, segmentation and stage estimation are AI-assisted "
            "and should not replace medical diagnosis."
        )
