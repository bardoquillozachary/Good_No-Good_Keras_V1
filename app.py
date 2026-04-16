import streamlit as st
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO
import os

# ── PAGE CONFIG & CSS ────────────────────────────────────────
st.set_page_config(
    page_title="Wire Heat-Shrink Inspector (YOLO)",
    layout="wide",
    page_icon="⚡"
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Inter:wght@400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0d1117; color: #e6edf3; }
  .stApp { background-color: #0d1117; }
  .inspector-title { font-family: 'Share Tech Mono', monospace; font-size: 1.6rem; color: #58a6ff; letter-spacing: 2px; border-bottom: 1px solid #21262d; padding-bottom: 0.5rem; margin-bottom: 1rem; }
  .badge-pass { display: inline-block; padding: 6px 18px; background: #0d2d1a; border: 1.5px solid #2ea043; border-radius: 6px; color: #3fb950; font-family: 'Share Tech Mono', monospace; font-size: 1.1rem; letter-spacing: 1px; }
  .badge-fail { display: inline-block; padding: 6px 18px; background: #2d0d0d; border: 1.5px solid #da3633; border-radius: 6px; color: #f85149; font-family: 'Share Tech Mono', monospace; font-size: 1.1rem; letter-spacing: 1px; }
  .metric-card { background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 14px 20px; margin-bottom: 8px; }
  .metric-label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
  .metric-value { font-family: 'Share Tech Mono', monospace; font-size: 1.4rem; color: #e6edf3; }
</style>
""", unsafe_allow_html=True)

# ── YOLO MODEL LOADER ────────────────────────────────────────
@st.cache_resource
def load_model():
    # Supports either the standard PyTorch file or the faster ONNX export
    for model_name in ["best.onnx", "best.pt", "wire_yolo.pt"]:
        if os.path.exists(model_name):
            st.sidebar.success(f"✅ Loaded YOLO Model: {model_name}")
            return YOLO(model_name, task='classify')
    
    st.error("⛔ No model found. Place your best.pt or best.onnx file next to app.py")
    return None

model = load_model()

# ── ROI EXTRACTOR (Kept to find the wire) ────────────────────
def get_wire_bbox(frame):
    """Finds the bounding box of the wire to crop the heat-shrink zone."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
    cm = np.mean(blurred[h//4:3*h//4, w//4:3*w//4])
    om = np.mean(blurred)
    mode = cv2.THRESH_BINARY_INV if cm < om else cv2.THRESH_BINARY
    _, thr = cv2.threshold(blurred, 0, 255, mode + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    closed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    wire_cnts = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 3000:
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / max(ch, 1)
            if 0.67 <= aspect <= 1.5:
                wire_cnts.append((cv2.contourArea(cnt), x, y, cw, ch, cw >= ch))
                
    if not wire_cnts: return None
    wire_cnts.sort(reverse=True)
    _, x, y, cw, ch, is_horiz = wire_cnts[0]
    return x, y, cw, ch, is_horiz

def extract_and_annotate(frame, target_size=224):
    """Crops the ROI and draws a box on the original frame so you can see what it found."""
    h, w = frame.shape[:2]
    bbox_data = get_wire_bbox(frame)
    annotated = frame.copy()
    
    if bbox_data:
        x, y, cw, ch, is_horiz = bbox_data
        if is_horiz:
            hs_x, hs_w = x + cw // 3, cw // 3
            pad_v = int(ch * 0.20)
            hs_y, hs_h = max(0, y - pad_v), min(h - max(0, y - pad_v), ch + 2*pad_v)
        else:
            hs_y, hs_h = y + ch // 3, ch // 3
            pad_h = int(cw * 0.20)
            hs_x, hs_w = max(0, x - pad_h), min(w - max(0, x - pad_h), cw + 2*pad_h)
            
        crop = frame[hs_y:hs_y+hs_h, hs_x:hs_x+hs_w]
        cv2.rectangle(annotated, (hs_x, hs_y), (hs_x+hs_w, hs_y+hs_h), (255, 200, 0), 2)
        
        if crop.size > 0:
            return cv2.resize(crop, (target_size, target_size)), annotated

    # Fallback to center crop if no wire found
    cx, cy, half = w//2, h//2, int(min(w, h) * 0.45 / 2)
    fallback = frame[max(0, cy-half):cy+half, max(0, cx-half):cx+half]
    cv2.rectangle(annotated, (max(0, cx-half), max(0, cy-half)), (cx+half, cy+half), (255, 200, 0), 2)
    return cv2.resize(fallback, (target_size, target_size)), annotated

# ── YOLO PREDICTION ──────────────────────────────────────────
def predict_yolo(roi_bgr):
    """Passes the crop to YOLO and returns the formatted results."""
    # YOLO automatically converts BGR to RGB internally during inference
    results = model(roi_bgr, verbose=False)
    
    # Extract classification info
    top_class_idx = results[0].probs.top1
    confidence = results[0].probs.top1conf.item()
    class_name = results[0].names[top_class_idx].lower()
    
    if "good" in class_name and "no" not in class_name:
        return "good", confidence, f"PASS ({confidence*100:.1f}%)"
    else:
        return "ng", confidence, f"FAIL ({confidence*100:.1f}%)"

# ── SIDEBAR & UI ─────────────────────────────────────────────
st.sidebar.markdown("## ⚙️ Controls")
mode = st.sidebar.radio("Mode", ["📂 Image Upload", "📹 Live Camera"], label_visibility="collapsed")
camera_index = st.sidebar.number_input("Camera Index", min_value=0, max_value=10, value=0, step=1)

st.markdown('<div class="inspector-title">⚡ YOLO WIRE INSPECTOR</div>', unsafe_allow_html=True)

if model is None: st.stop()

# ── IMAGE UPLOAD MODE ────────────────────────────────────────
if "Image Upload" in mode:
    col_left, col_right = st.columns([1.1, 1], gap="large")
    with col_left:
        uploaded = st.file_uploader("Drop a wire image here", type=["jpg", "jpeg", "png"])

    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        roi, annotated_frame = extract_and_annotate(frame)
        status, conf, text = predict_yolo(roi)
        
        # UI Rendering
        with col_left:
            st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption="Target Area Detected", use_column_width=True)
            
        with col_right:
            st.image(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), caption="YOLO Input (ROI)", width=224)
            st.markdown("---")
            
            badge_cls = "badge-pass" if status == "good" else "badge-fail"
            icon = "✅" if status == "good" else "❌"
            st.markdown(f'<div class="{badge_cls}">{icon} &nbsp; {text}</div><br>', unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">YOLO Confidence</div>
              <div class="metric-value">{conf:.4f}</div>
            </div>""", unsafe_allow_html=True)

# ── LIVE CAMERA MODE ─────────────────────────────────────────
else:
    col_feed, col_status = st.columns([1.6, 1], gap="large")
    with col_feed:
        frame_placeholder = st.empty()
        stop_btn = st.button("⏹ Stop Camera", type="primary")
        
    with col_status:
        result_placeholder = st.empty()
        conf_placeholder = st.empty()
        roi_placeholder = st.empty()

    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        st.error(f"Cannot open camera {camera_index}")
    else:
        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret: break
            
            roi, annotated_frame = extract_and_annotate(frame)
            status, conf, text = predict_yolo(roi)
            
            # Draw result directly on the live frame
            color = (0, 255, 0) if status == "good" else (0, 0, 255)
            cv2.putText(annotated_frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            frame_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
            roi_placeholder.image(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), caption="YOLO Input", width=150)
            
            badge_cls = "badge-pass" if status == "good" else "badge-fail"
            icon = "✅" if status == "good" else "❌"
            result_placeholder.markdown(f'<div class="{badge_cls}">{icon} &nbsp; {text}</div>', unsafe_allow_html=True)
            
            conf_placeholder.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">YOLO Confidence</div>
              <div class="metric-value">{conf:.4f}</div>
            </div>""", unsafe_allow_html=True)

        cap.release()
        st.info("Camera stopped.")