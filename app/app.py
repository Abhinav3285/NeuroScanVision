import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
from datetime import datetime
import cv2

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor Detector",
    layout="wide",
    page_icon="🧠"
)

# -------------------------------------------------
# Load YOLOv8 Model
# -------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("../results/brain_tumor_run/weights/best.pt")

model = load_model()

# -------------------------------------------------
# Grad-CAM like Heatmap for YOLOv8 (SAFE VERSION)
# -------------------------------------------------
def generate_yolo_cam(image_np, results):
    heatmap = np.zeros(image_np.shape[:2], dtype=np.float32)

    if results[0].boxes is None:
        return image_np

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        heatmap[y1:y2, x1:x2] += float(box.conf[0])

    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)
    return overlay

# -------------------------------------------------
# PDF Generator
# -------------------------------------------------
def generate_pdf(image_pil, report_text):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, height - 40, "AI Brain Tumor Detection Report")

    c.setFont("Helvetica", 10)
    c.drawString(
        40,
        height - 60,
        f"Generated on: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
    )

    img_buffer = io.BytesIO()
    image_pil.save(img_buffer, format="PNG")
    img_buffer.seek(0)
    img_reader = ImageReader(img_buffer)

    c.drawImage(img_reader, 40, height - 380, width=300, height=300)

    text = c.beginText(40, height - 420)
    text.setFont("Helvetica", 11)

    for line in report_text.split("\n"):
        text.textLine(line)

    c.drawText(text)
    c.showPage()
    c.save()

    buffer.seek(0)
    return buffer

# -------------------------------------------------
# UI Header
# -------------------------------------------------
st.title("🧠 AI Brain Tumor Detection System")
st.caption("⚠️ AI-assisted prediction. Not for medical diagnosis.")

with st.sidebar:
    st.header("🛠️ Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# -------------------------------------------------
# Upload Images
# -------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload MRI scan(s)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# -------------------------------------------------
# Process Images
# -------------------------------------------------
if uploaded_files:
    for uploaded_file in uploaded_files:
        st.divider()
        st.subheader(f"📄 File: {uploaded_file.name}")

        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🖼️ Original MRI")
            st.image(image, use_container_width=True)
            st.write(f"Resolution: {image.size[0]} × {image.size[1]}")
            st.write("Slice Type: Axial (assumed)")

        with col2:
            st.subheader("🎯 YOLO Detection")
            results = model.predict(img_np, conf=conf_threshold)
            detected_img = Image.fromarray(results[0].plot())
            st.image(detected_img, use_container_width=True)

        # -------------------------------------------------
        # Grad-CAM Visualization
        # -------------------------------------------------
        st.subheader("🔥 Explainable AI (Grad-CAM Visualization)")
        cam_overlay = generate_yolo_cam(img_np, results)
        st.image(cam_overlay, caption="Highlighted Regions Influencing Prediction", use_container_width=True)

        # -------------------------------------------------
        # Tumor Details
        # -------------------------------------------------
        st.subheader("🧪 Tumor Details")
        boxes = results[0].boxes
        report_lines = []

        if boxes is not None and len(boxes) > 0:
            st.success(f"Detected {len(boxes)} tumor(s)")

            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                confidence = float(box.conf[0])

                x1, y1, x2, y2 = box.xyxy[0]
                area = int((x2 - x1) * (y2 - y1))

                if area < 5000:
                    stage = "Stage I (Low)"
                    risk = "Low Risk"
                elif area < 15000:
                    stage = "Stage II (Moderate)"
                    risk = "Medium Risk"
                else:
                    stage = "Stage III (High)"
                    risk = "High Risk"

                with st.container(border=True):
                    st.markdown(f"### Tumor {i+1}")
                    st.write(f"Type: **{label.upper()}**")
                    st.write(f"Size: **{area:,} px²**")
                    st.write(f"Stage: **{stage}**")
                    st.write(f"Risk: **{risk}**")
                    st.progress(confidence)

                report_lines.append(
                    f"Tumor {i+1}: {label} | Conf: {confidence:.2%} | "
                    f"Size: {area}px² | {stage} | {risk}"
                )

            st.info(
                "⚠️ Stage estimation is AI-assisted and based on tumor size only. "
                "Not a medical diagnosis."
            )

            # -------------------------------------------------
            # PDF Report
            # -------------------------------------------------
            pdf = generate_pdf(detected_img, "\n".join(report_lines))
            st.download_button(
                "📄 Download PDF Medical Report",
                data=pdf,
                file_name="Brain_Tumor_Report.pdf",
                mime="application/pdf"
            )

        else:
            st.warning("No tumors detected.")
