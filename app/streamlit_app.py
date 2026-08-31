import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import threading

# ------------------ CONFIG ------------------
MODEL_PATH = "models/resnet18_best.pth"
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CONFIDENCE_THRESHOLD = 0.60

device = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(
    page_title="Visual Quality Inspector",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 45%, #4c1d95 100%);
    padding: 3rem 2.5rem;
    border-radius: 24px;
    margin-bottom: 2rem;
    box-shadow: 0 20px 50px rgba(76, 29, 149, 0.25);
    position: relative;
    overflow: hidden;
    animation: fadeIn 0.8s ease;
}

.main-header::before {
    content: "";
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
}

.main-header h1 {
    color: white;
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}

.main-header p {
    color: rgba(255,255,255,0.75);
    font-size: 1rem;
    font-weight: 400;
    margin-top: 0.6rem;
    letter-spacing: 0.2px;
}

.header-badge {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    color: rgba(255,255,255,0.9);
    padding: 0.3rem 0.9rem;
    border-radius: 50px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 1rem;
    border: 1px solid rgba(255,255,255,0.15);
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-10px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

.glass-card {
    background: rgba(255, 255, 255, 0.035);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 18px;
    padding: 1.6rem 1.5rem;
    margin-bottom: 1rem;
    animation: fadeInUp 0.5s ease;
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
}

.glass-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 14px 32px rgba(0,0,0,0.22);
    border-color: rgba(124, 58, 237, 0.35);
}

.card-icon {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    background: linear-gradient(135deg, #7c3aed, #4c1d95);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    margin-bottom: 0.8rem;
}

.glass-card h3 {
    font-size: 1.05rem;
    font-weight: 600;
    margin: 0 0 0.5rem 0;
    color: rgba(255,255,255,0.92);
}

.glass-card p {
    font-size: 0.9rem;
    line-height: 1.55;
    color: rgba(255,255,255,0.6);
    margin: 0;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.55rem 1.3rem;
    border-radius: 50px;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.2px;
}

.status-defect {
    background: rgba(239, 68, 68, 0.12);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #f87171;
    animation: dotPulse 1.6s infinite;
}

@keyframes dotPulse {
    0% { box-shadow: 0 0 0 0 rgba(248,113,113,0.5); }
    70% { box-shadow: 0 0 0 8px rgba(248,113,113,0); }
    100% { box-shadow: 0 0 0 0 rgba(248,113,113,0); }
}

.metric-container {
    text-align: center;
    padding: 1.4rem 1rem;
    background: rgba(255,255,255,0.03);
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.06);
    height: 100%;
}

.metric-value {
    font-size: 1.7rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-label {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.45);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-top: 0.4rem;
    font-weight: 500;
}

.section-title {
    font-size: 1.05rem;
    font-weight: 600;
    margin-bottom: 0.9rem;
    color: rgba(255,255,255,0.85);
    letter-spacing: 0.2px;
}

.section-title::before {
    content: "";
    display: inline-block;
    width: 3px;
    height: 14px;
    background: linear-gradient(180deg, #a78bfa, #7c3aed);
    margin-right: 8px;
    border-radius: 2px;
    vertical-align: middle;
}

[data-testid="stFileUploader"] {
    border: 1.5px dashed rgba(124, 58, 237, 0.35);
    border-radius: 16px;
    padding: 0.8rem;
    transition: all 0.25s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(124, 58, 237, 0.6);
    background: rgba(124, 58, 237, 0.04);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0;
    padding: 10px 22px;
    font-weight: 500;
    font-size: 0.92rem;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #a78bfa, #7c3aed);
}

.pipeline-strip {
    text-align: center;
    font-weight: 500;
    font-size: 0.95rem;
    letter-spacing: 0.3px;
    color: rgba(255,255,255,0.7);
    line-height: 2.2;
}

.footer-note {
    text-align: center;
    color: rgba(255,255,255,0.35);
    font-size: 0.8rem;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(255,255,255,0.07);
}
</style>
""", unsafe_allow_html=True)

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint['class_names']

    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, class_names, checkpoint

model, class_names, checkpoint = load_model()

target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

def denormalize(tensor):
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img

def predict_and_explain(pil_image):
    input_tensor_raw = transform(pil_image.convert("RGB"))
    input_tensor = input_tensor_raw.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    rgb_img = denormalize(input_tensor_raw)
    overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    all_probs = {class_names[i]: probs[i].item() for i in range(len(class_names))}

    return {
        "predicted_class": class_names[pred_idx],
        "confidence": confidence,
        "overlay": overlay,
        "heatmap": grayscale_cam,
        "all_probabilities": all_probs
    }

# ------------------ HEADER ------------------
st.markdown("""
<div class="main-header">
    <div class="header-badge">Explainable AI · Deep Learning</div>
    <h1>Visual Quality Inspector</h1>
    <p>Automated defect detection with transparent, visual reasoning behind every prediction</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Overview", "Inspection", "Model Details"])

# ---------- TAB 1: HOME ----------
with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="glass-card">
            <div class="card-icon">◎</div>
            <h3>What It Does</h3>
            <p>Analyzes product surface images and identifies manufacturing defects 
            using a fine-tuned ResNet18 model.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="glass-card">
            <div class="card-icon">◈</div>
            <h3>Why It Matters</h3>
            <p>Manual inspection is slow and inconsistent. This system offers fast, 
            explainable screening to support quality control teams.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="glass-card">
            <div class="card-icon">◆</div>
            <h3>Explainable AI</h3>
            <p>Every prediction is paired with a Grad-CAM heatmap, showing exactly 
            where the model focused to make its decision.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Inspection Pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card pipeline-strip">
    Product Image &nbsp;→&nbsp; Preprocessing &nbsp;→&nbsp; CNN Model &nbsp;→&nbsp; 
    Prediction &nbsp;→&nbsp; Confidence Score &nbsp;→&nbsp; Grad-CAM &nbsp;→&nbsp; Report
    </div>
    """, unsafe_allow_html=True)

    st.warning("This is a portfolio/educational prototype, not a certified industrial safety system.")

# ---------- TAB 2: IMAGE INSPECTION ----------
with tab2:
    st.markdown('<div class="section-title">Provide a Product Image</div>', unsafe_allow_html=True)

    input_method = st.radio(
        "Choose input method",
        ["Upload Image", "Take Photo", "Live Camera Feed"],
        horizontal=True,
        label_visibility="collapsed"
    )

    uploaded_file = None

    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "bmp"])

    elif input_method == "Take Photo":
        uploaded_file = st.camera_input("Take a photo of the product surface")

    else:  # Live Camera Feed
            st.caption("Live feed shows the raw camera stream. Click Capture & Analyze to run inspection on the current frame.")
            st.caption("⚡ Experimental feature — real-time WebRTC connection reliability depends on your network. If it doesn't connect, use 'Take Photo' instead for the same live-camera inspection experience.")

        lock = threading.Lock()

        class VideoProcessor(VideoProcessorBase):
            def __init__(self):
                self.latest_frame = None

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                with lock:
                    self.latest_frame = img.copy()
                return av.VideoFrame.from_ndarray(img, format="bgr24")

        RTC_CONFIGURATION = {
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {
                    "urls": ["turn:openrelay.metered.ca:80"],
                    "username": "openrelayproject",
                    "credential": "openrelayproject",
                },
                {
                    "urls": ["turn:openrelay.metered.ca:443"],
                    "username": "openrelayproject",
                    "credential": "openrelayproject",
                },
                {
                    "urls": ["turn:openrelay.metered.ca:443?transport=tcp"],
                    "username": "openrelayproject",
                    "credential": "openrelayproject",
                },
            ]
        }

        ctx = webrtc_streamer(
            key="live-inspection",
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration=RTC_CONFIGURATION,
        )

        if st.button("Capture & Analyze Current Frame"):
            if ctx.video_processor:
                with lock:
                    frame = ctx.video_processor.latest_frame
                if frame is not None:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    uploaded_file = Image.fromarray(rgb_frame)
                else:
                    st.warning("No frame captured yet. Make sure the camera stream is running.")
            else:
                st.warning("Camera stream not active. Please allow camera access above.")

    if uploaded_file is not None:
        try:
            if isinstance(uploaded_file, Image.Image):
                image = uploaded_file
            else:
                image = Image.open(uploaded_file)

            with st.spinner("Running inspection..."):
                result = predict_and_explain(image)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="section-title">Original Image</div>', unsafe_allow_html=True)
                st.image(image, use_container_width=True)
            with col2:
                st.markdown('<div class="section-title">Grad-CAM Explanation</div>', unsafe_allow_html=True)
                st.image(result["overlay"], use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)

            confidence = result["confidence"]
            predicted_class = result["predicted_class"]

            if confidence < CONFIDENCE_THRESHOLD:
                st.warning(f"Low-confidence prediction ({confidence:.1%}). Manual review recommended.")

            col3, col4, col5 = st.columns(3)
            with col3:
                st.markdown(f"""
                <div class="metric-container">
                    <span class="status-badge status-defect"><span class="status-dot"></span>Defect Detected</span>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{predicted_class}</div>
                    <div class="metric-label">Defect Type</div>
                </div>
                """, unsafe_allow_html=True)
            with col5:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{confidence:.1%}</div>
                    <div class="metric-label">Confidence</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">Class Probabilities</div>', unsafe_allow_html=True)

            sorted_probs = dict(sorted(result["all_probabilities"].items(), key=lambda x: x[1], reverse=True))
            for cls, prob in sorted_probs.items():
                c1, c2 = st.columns([1, 4])
                c1.write(f"**{cls}**")
                c2.progress(prob, text=f"{prob:.2%}")

            st.info("""
            **Note on Grad-CAM:** This heatmap shows the approximate region the model 
            focused on. It is not a precise pixel-level defect boundary — treat it as a 
            general area of interest, not a guaranteed localization.
            """)

        except Exception as e:
            st.error(f"Error processing image: {e}")
    else:
        st.info("Provide a product image above to begin inspection.")

# ---------- TAB 3: MODEL INFO ----------
with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="glass-card">
            <h3>Architecture</h3>
            <p><b>Model:</b> {checkpoint.get('architecture', 'ResNet18')}<br>
            <b>Input size:</b> {checkpoint.get('img_size', IMG_SIZE)}x{checkpoint.get('img_size', IMG_SIZE)}<br>
            <b>Classes:</b> {', '.join(class_names)}<br>
            <b>Device:</b> {device}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        test_acc = checkpoint.get('test_accuracy', 0)
        test_f1 = checkpoint.get('test_f1', 0)
        st.markdown(f"""
        <div class="glass-card">
            <h3>Test Performance</h3>
            <p><b>Test Accuracy:</b> {test_acc:.2%}<br>
            <b>Test F1-score:</b> {test_f1:.4f}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <h3>Dataset</h3>
        <p><b>Dataset:</b> NEU Surface Defect Database (NEU-DET)<br>
        <b>Total images:</b> 1800 (6 classes, 300 images each)<br>
        <b>Split:</b> 70% train / 15% validation / 15% test</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer-note">
    Built with PyTorch, Streamlit & Grad-CAM · Visual Quality Inspector Portfolio Project
</div>
""", unsafe_allow_html=True)
