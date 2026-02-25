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

/* ===== GLOBAL TEXT STYLING ===== */
/* Premium Background */
.stApp {
    background: linear-gradient(135deg, #0a0e27 0%, #1a1a3a 25%, #2d1b3d 50%, #1a1a2e 75%, #0f0f1e 100%);
    color: #e8e8e8;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* All text elements - Light color */
body, p, span, div, label, a {
    color: #e8e8e8 !important;
}

/* Paragraph text */
p {
    color: #d9d9d9 !important;
    font-size: 15px;
    line-height: 1.6;
}

/* Links */
a {
    color: #a89968 !important;
    text-decoration: none;
    transition: color 0.3s ease;
}

a:hover {
    color: #d4af37 !important;
}

/* ===== SIDEBAR ===== */
/* Premium Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-right: 3px solid #d4af37;
    box-shadow: inset -10px 0 30px rgba(212, 175, 55, 0.1);
}

/* Sidebar text - Light color */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    color: #d9d9d9 !important;
}

/* Menu title - Elegant Gold */
.menu-title {
    font-size: 24px;
    font-weight: 700;
    color: #d4af37 !important;
    text-shadow: 0 2px 8px rgba(212, 175, 55, 0.3);
    margin-bottom: 25px;
    letter-spacing: 1px;
}

/* ===== RADIO BUTTONS ===== */
/* Radio labels - Premium */
.stRadio label {
    font-size: 16px !important;
    color: #c9c9c9 !important;
    padding: 12px 16px !important;
    border-radius: 8px !important;
    transition: all 0.4s ease !important;
    font-weight: 500 !important;
    border-left: 3px solid transparent !important;
}

/* Hover effect - Elegant */
.stRadio label:hover {
    background: rgba(212, 175, 55, 0.15) !important;
    border-left: 3px solid #d4af37 !important;
    color: #d4af37 !important;
    box-shadow: 0 4px 12px rgba(212, 175, 55, 0.2) !important;
}

/* Selected state - Premium Gold */
input[type="radio"]:checked + div {
    background: rgba(212, 175, 55, 0.25) !important;
    border-left: 3px solid #d4af37 !important;
    box-shadow: 0 6px 20px rgba(212, 175, 55, 0.3) !important;
}

/* ===== HEADINGS ===== */
h1, h2, h3, h4, h5, h6 {
    color: #d4af37 !important;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
}

h1 { font-size: 28px; }
h2 { font-size: 24px; }
h3 { font-size: 20px; }

/* ===== BUTTONS ===== */
.stButton > button {
    background: linear-gradient(135deg, #d4af37, #e5c158);
    color: #1a1a2e !important;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(212, 175, 55, 0.3);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #e5c158, #d4af37);
    box-shadow: 0 6px 20px rgba(212, 175, 55, 0.5);
    transform: translateY(-2px);
}

/* ===== FILE UPLOADER ===== */
.stFileUploader {
    border: 2px dashed #d4af37;
    border-radius: 10px;
    padding: 20px;
}

.stFileUploader label {
    color: #d9d9d9 !important;
    font-weight: 500;
}

/* ===== METRICS ===== */
.stMetric {
    background: linear-gradient(135deg, rgba(212, 175, 55, 0.1), rgba(212, 175, 55, 0.05));
    border-left: 4px solid #d4af37;
    border-radius: 8px;
    padding: 15px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}

/* Metric labels and values - Light */
.stMetric label,
.stMetric div {
    color: #d9d9d9 !important;
}

.metric-label {
    color: #c9c9c9 !important;
}

.metric-value {
    color: #d4af37 !important;
    font-weight: 600;
}

/* ===== INFO/SUCCESS/WARNING BOXES ===== */
.stInfo, .stSuccess, .stWarning, .stError {
    border-radius: 8px;
}

.stInfo {
    background: linear-gradient(90deg, rgba(79, 172, 254, 0.15), transparent);
    border-left: 4px solid #4facfe;
}

.stInfo p, .stInfo span {
    color: #b3e5fc !important;
}

.stSuccess {
    background: linear-gradient(90deg, rgba(76, 212, 139, 0.15), transparent);
    border-left: 4px solid #4cd48b;
}

.stSuccess p, .stSuccess span {
    color: #a8e6d3 !important;
}

.stWarning {
    background: linear-gradient(90deg, rgba(255, 193, 7, 0.15), transparent);
    border-left: 4px solid #ffc107;
}

.stWarning p, .stWarning span {
    color: #ffe082 !important;
}

.stError {
    background: linear-gradient(90deg, rgba(244, 67, 54, 0.15), transparent);
    border-left: 4px solid #f44336;
}

.stError p, .stError span {
    color: #ef9a9a !important;
}

/* ===== TEXT INPUTS & SELECTS ===== */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div > select,
.stTextArea > div > div > textarea {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid #d4af37 !important;
    color: #e8e8e8 !important;
    border-radius: 6px;
}

.stTextInput > div > div > input::placeholder,
.stNumberInput > div > div > input::placeholder,
.stTextArea > div > div > textarea::placeholder {
    color: #888888 !important;
}

.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stSelectbox > div > div > select:focus,
.stTextArea > div > div > textarea:focus {
    border: 2px solid #d4af37 !important;
    box-shadow: 0 0 10px rgba(212, 175, 55, 0.3) !important;
}

/* Input labels */
.stTextInput label,
.stNumberInput label,
.stSelectbox label,
.stTextArea label {
    color: #d9d9d9 !important;
    font-weight: 500;
}

/* ===== CHECKBOX & OTHER INPUTS ===== */
.stCheckbox label {
    color: #d9d9d9 !important;
    font-size: 16px !important;
}

.stCheckbox label:hover {
    color: #d4af37 !important;
}

/* ===== DIVIDERS ===== */
hr {
    border: 0;
    border-top: 1px solid rgba(212, 175, 55, 0.3);
    margin: 30px 0;
}

/* ===== IMAGE CONTAINERS ===== */
.stImage {
    border-radius: 10px;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
}

/* ===== CODE BLOCKS ===== */
code {
    background: rgba(212, 175, 55, 0.1);
    color: #90ee90 !important;
    border-radius: 4px;
    padding: 2px 6px;
}

/* ===== TABS ===== */
.stTabs [role="tablist"] button {
    color: #d9d9d9 !important;
}

.stTabs [role="tablist"] button[aria-selected="true"] {
    color: #d4af37 !important;
    border-bottom: 3px solid #d4af37;
}

.stTabs [role="tablist"] button:hover {
    color: #d4af37 !important;
}

/* ===== COLUMN LABELS ===== */
.stColumn label, .stColumn span {
    color: #d9d9d9 !important;
}

/* ===== EXPANDER ===== */
.stExpander summary {
    color: #d9d9d9 !important;
    font-weight: 600;
}

.stExpander summary:hover {
    color: #d4af37 !important;
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