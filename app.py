import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO

st.set_page_config(page_title="Smart Waste AI", layout="wide")

# ================= COOL NAVIGATION CSS =================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#000428,#004e92);
    border-right: 2px solid cyan;
    box-shadow: 0 0 20px cyan;
}

/* Menu title */
.menu-title {
    font-size: 22px;
    font-weight: bold;
    color: cyan;
    text-shadow: 0 0 12px cyan;
    margin-bottom: 15px;
}

/* Radio labels */
.stRadio label {
    font-size: 18px !important;
    color: #e0f7ff !important;
    padding: 10px;
    border-radius: 10px;
    transition: all 0.3s ease;
}

/* Hover glow */
.stRadio label:hover {
    background: rgba(0,255,255,0.2);
    box-shadow: 0 0 10px cyan;
}

/* Selected glow */
input[type="radio"]:checked + div {
    background: rgba(0,255,255,0.3);
    border-left: 4px solid cyan;
    box-shadow: 0 0 10px cyan;
}

</style>
""", unsafe_allow_html=True)

# ================= LOAD MODELS =================

@st.cache_resource
def load_models():
    organic = tf.keras.models.load_model("organic_model.keras")
    material = tf.keras.models.load_model("garbage_model_v2.keras")
    yolo = YOLO("yolov8n.pt")
    return organic, material, yolo

organic_model, material_model, yolo_model = load_models()

material_classes = ["cardboard","glass","metal","paper","plastic","trash"]
organic_classes = ["organic","recyclable"]

recycle_map = {
    "cardboard":"♻ Recyclable",
    "glass":"♻ Recyclable",
    "metal":"♻ Recyclable",
    "paper":"♻ Recyclable",
    "plastic":"♻ Recyclable",
    "trash":"🚫 Not Recyclable"
}

def preprocess(img):
    img = img.resize((224,224))
    arr = np.array(img)/255.0
    return np.expand_dims(arr, axis=0)

def predict_organic(img):
    preds = organic_model.predict(preprocess(img), verbose=0)
    return organic_classes[np.argmax(preds)]

def predict_material(img):
    preds = material_model.predict(preprocess(img), verbose=0)
    return material_classes[np.argmax(preds)]

# ================= SIDEBAR =================

st.sidebar.markdown('<div class="menu-title">⚡ AI Control Panel</div>', unsafe_allow_html=True)

page = st.sidebar.radio("",
[
    "🧠 Neural Scan — Detection",
    "🎥 Vision Feed — Live",
    "📊 System Core — Dashboard",
    "📈 Diagnostics — Evaluation"
])

# ================= DETECTION =================

if page == "🧠 Neural Scan — Detection":

    st.title("♻  Waste classification")

    uploaded = st.file_uploader("Upload image")

    if uploaded:

        image = Image.open(uploaded).convert("RGB")
        st.image(image, width=400)

        frame = np.array(image)
        frame = cv2.resize(frame,(640,640))

        results = yolo_model(frame, conf=0.25)[0]

        count = 0

        for box in results.boxes:

            x1,y1,x2,y2 = map(int, box.xyxy[0])

            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            crop_img = Image.fromarray(crop)

            org = predict_organic(crop_img)

            if org == "organic":
                label = "Organic"
                color = (0,255,0)
            else:
                mat = predict_material(crop_img)
                label = f"{mat} | {recycle_map[mat]}"
                color = (255,255,0)

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,label,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

            count += 1

        st.image(frame, channels="RGB")
        st.success(f"Detected objects: {count}")

# ================= LIVE =================

elif page == "🎥 Vision Feed — Live":

    st.title("🎥 Real-Time Detection")

    start = st.button("▶ Start")
    stop = st.button("⏹ Stop")

    if "run" not in st.session_state:
        st.session_state.run = False

    if start:
        st.session_state.run = True
    if stop:
        st.session_state.run = False

    frame_placeholder = st.empty()

    cap = cv2.VideoCapture(0)

    while st.session_state.run:

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame,(640,480))

        results = yolo_model(frame, conf=0.25)[0]

        count = 0

        for box in results.boxes:

            x1,y1,x2,y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            crop_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

            org = predict_organic(crop_img)

            if org == "organic":
                label = "Organic"
                color = (0,255,0)
            else:
                mat = predict_material(crop_img)
                label = mat
                color = (0,255,255)

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,label,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

            count += 1

        cv2.putText(frame,f"Count: {count}",
                    (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        frame_placeholder.image(frame, channels="BGR")

    cap.release()

# ================= DASHBOARD =================

elif page == "📊 System Core — Dashboard":

    st.title("📊 AI Dashboard")

    c1,c2,c3 = st.columns(3)

    c1.metric("Models","YOLO + TensorFlow")
    c2.metric("Pipeline","Hierarchical")
    c3.metric("Status","Running")

    st.info("Real-time waste monitoring system.")

# ================= EVALUATION =================

elif page == "📈 Diagnostics — Evaluation":

    st.title("📈 Model Diagnostics")

    st.write("""
✔ YOLO object detection  
✔ Organic classification  
✔ Material classification  
✔ Real-time inference  
✔ Recycling decision  
""")

    st.success("System ready for demo.")