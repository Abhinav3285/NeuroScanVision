import streamlit as st
from ultralytics import YOLO
import PIL.Image
import numpy as np
import io

# 1. Page Configuration
st.set_page_config(page_title="Brain Tumor Detector", layout="wide", page_icon="🧠")

# 2. Load the Model
@st.cache_resource
def load_model():
    return YOLO('../results/brain_tumor_run/weights/best.pt')

model = load_model()

# 3. Sidebar and Header
st.title("🧠 AI Brain Tumor Detection System")
with st.sidebar:
    st.header("🛠️ Settings")
    conf_threshold = st.slider("AI Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# 4. File Uploader
uploaded_file = st.file_uploader("Upload an MRI scan", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file)
    img_array = np.array(image)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🖼️ Original MRI")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🎯 AI Detection Result")
        results = model.predict(img_array, conf=conf_threshold)
        res_plotted = results[0].plot()
        
        # Convert the result (numpy array) back to a PIL image for display/download
        res_image = PIL.Image.fromarray(res_plotted.astype('uint8'))
        st.image(res_image, caption="Detected Tumors", use_container_width=True)

        # --- NEW: PREPARE DOWNLOAD BUTTON ---
        # We must save the image to a "virtual folder" (buffer) so the user can download it
        buf = io.BytesIO()
        res_image.save(buf, format="JPEG")
        byte_im = buf.getvalue()

        st.download_button(
            label="📥 Download Result Image",
            data=byte_im,
            file_name="tumor_detection_result.jpg",
            mime="image/jpeg"
        )

    # 5. Text Summary
    st.divider()
    boxes = results[0].boxes
    if len(boxes) > 0:
        st.success(f"Found {len(boxes)} tumor(s)!")
        for box in boxes:
            label = model.names[int(box.cls[0])]
            st.write(f"🔍 **Type:** {label.upper()} | **Certainty:** {float(box.conf[0]):.2%}")
    else:
        st.warning("No tumors detected.")