import os
import streamlit as st
import cv2
import tensorflow as tf
from PIL import Image
import numpy as np

# ── GLOBAL KERAS PATCH ───────────────────────────────────────
# Keras 3 ignores custom_objects for built-in layers. 
# Instead, we intercept the Dense layer's initialization globally
# and surgically remove 'quantization_config' if it exists.
_original_dense_init = tf.keras.layers.Dense.__init__

def _patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    _original_dense_init(self, *args, **kwargs)

tf.keras.layers.Dense.__init__ = _patched_dense_init
# ─────────────────────────────────────────────────────────────

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Wire Heat-Shrink Inspector",
    layout="wide",
    page_icon="🛡️"
)

# ... (Keep your CSS styling block here) ...

# ── MODEL LOADER ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    for name in ["wire_model_full.keras", "wire_model_legacy.h5"]:
        if os.path.exists(name):
            # compile=False skips loading optimizers (safer for inference)
            m = tf.keras.models.load_model(name, compile=False)
            st.sidebar.success(f"✅ Loaded {name}")
            return m
            
    st.error("⛔ No model file found. Place your .keras file next to app.py")
    return None

# ... (Keep the rest of your app code exactly the same) ...

model = load_model()

# ── POLARITY HELPER ───────────────────────────────────────────
def get_threshold_mode(gray):
    """Auto-detect dark-on-light vs light-on-dark."""
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    cm = np.mean(blurred[h//4:3*h//4, w//4:3*w//4])
    om = np.mean(blurred)
    if cm < om:
        return cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    return cv2.THRESH_BINARY + cv2.THRESH_OTSU

# ── BACKGROUND ISOLATION ─────────────────────────────────────
def isolate_foreground(frame, blur_amount=21, grabcut_iters=5):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    mode = get_threshold_mode(gray)
    _, thr = cv2.threshold(blurred, 0, 255, mode)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    thr    = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rect = None
    if contours:
        areas = [(cv2.contourArea(c), c) for c in contours if cv2.contourArea(c) > 2000]
        if areas:
            areas.sort(reverse=True)
            _, best = areas[0]
            x, y, cw, ch = cv2.boundingRect(best)
            pad  = 20
            rect = (max(0, x-pad), max(0, y-pad),
                    min(w, cw+2*pad), min(h, ch+2*pad))

    if rect and rect[2] > 10 and rect[3] > 10:
        gc_mask   = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(frame, gc_mask, rect, bgd_model, fgd_model,
                        grabcut_iters, cv2.GC_INIT_WITH_RECT)
            fg_mask = np.where(
                (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
                255, 0).astype(np.uint8)
        except Exception:
            fg_mask = thr
    else:
        return frame.copy(), np.ones((h, w), np.uint8) * 255

    k2      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, k2, iterations=3)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  k2, iterations=1)
    soft    = cv2.GaussianBlur(fg_mask, (blur_amount, blur_amount), 0)
    mask_f  = soft.astype(np.float32) / 255.0
    mask_f  = np.stack([mask_f] * 3, axis=-1)
    dark_bg = (frame.astype(np.float32) * 0.08).astype(np.uint8)
    isolated = (frame.astype(np.float32) * mask_f +
                dark_bg.astype(np.float32) * (1 - mask_f)).astype(np.uint8)
    return isolated, fg_mask

# ── HEAT SHRINK ROI ───────────────────────────────────────────
def find_wire_bbox(frame):
    """Return (x, y, w, h, is_horizontal) of the largest wire contour, or None."""
    h, w = frame.shape[:2]
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    mode    = get_threshold_mode(gray)
    _, thr  = cv2.threshold(blurred, 0, 255, mode)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    closed  = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    wire_cnts = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 3000:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / max(ch, 1)
        if aspect > 1.5 or aspect < 0.67:
            wire_cnts.append((area, x, y, cw, ch))
    if not wire_cnts:
        return None
    wire_cnts.sort(reverse=True)
    _, x, y, cw, ch = wire_cnts[0]
    return x, y, cw, ch, (cw >= ch)

def extract_heatshrink_roi(frame, target_size=224):
    if frame is None or frame.size == 0:
        return None
    h, w   = frame.shape[:2]
    result = find_wire_bbox(frame)

    if result is not None:
        x, y, cw, ch, is_horiz = result
        if is_horiz:
            hs_x  = x + cw // 3
            hs_w  = cw // 3
            pad_v = int(ch * 0.20)
            hs_y  = max(0, y - pad_v)
            hs_h  = min(h - hs_y, ch + 2*pad_v)
        else:
            hs_y  = y + ch // 3
            hs_h  = ch // 3
            pad_h = int(cw * 0.20)
            hs_x  = max(0, x - pad_h)
            hs_w  = min(w - hs_x, cw + 2*pad_h)
        crop = frame[hs_y:hs_y+hs_h, hs_x:hs_x+hs_w]
        if crop.size > 0:
            return cv2.resize(crop, (target_size, target_size))

    cx, cy   = w//2, h//2
    half     = int(min(w, h) * 0.45 / 2)
    fallback = frame[max(0, cy-half):cy+half, max(0, cx-half):cx+half]
    return cv2.resize(fallback, (target_size, target_size)) \
           if fallback.size > 0 else None

# ── NG ZONE HIGHLIGHTER ───────────────────────────────────────
def highlight_ng_zones(frame, sensitivity, num_tiles=6):
    """
    Divide the wire's heat-shrink zone into num_tiles horizontal strips.
    Run the model on each strip independently.
    Draw a red semi-transparent overlay on strips that score >= sensitivity.
    Returns the annotated frame and a list of (score, is_ng) per tile.
    """
    if model is None:
        return frame, []

    result = find_wire_bbox(frame)
    if result is None:
        return frame, []

    h_img, w_img = frame.shape[:2]
    x, y, cw, ch, is_horiz = result
    annotated = frame.copy()
    tile_results = []

    if is_horiz:
        # Divide the heat-shrink zone (middle third) into vertical strips
        hs_x = x + cw // 3
        hs_w = cw // 3
        pad_v = int(ch * 0.20)
        hs_y  = max(0, y - pad_v)
        hs_h  = min(h_img - hs_y, ch + 2*pad_v)
        tile_w = max(1, hs_w // num_tiles)

        for i in range(num_tiles):
            tx  = hs_x + i * tile_w
            tw  = tile_w if i < num_tiles - 1 else (hs_x + hs_w - tx)
            if tw <= 0:
                continue
            tile = frame[hs_y:hs_y+hs_h, tx:tx+tw]
            if tile.size == 0:
                continue
            tile_resized = cv2.resize(
                cv2.cvtColor(tile, cv2.COLOR_BGR2RGB), (224, 224))
            inp   = np.expand_dims(tile_resized.astype(np.float32)/255.0, 0)
            score = float(model.predict(inp, verbose=0)[0][0])
            is_ng = score >= sensitivity
            tile_results.append((score, is_ng))

            if is_ng:
                # Red semi-transparent overlay on NG tile
                overlay = annotated.copy()
                cv2.rectangle(overlay, (tx, hs_y),
                              (tx+tw, hs_y+hs_h), (0, 0, 220), -1)
                cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)
                # Red border
                cv2.rectangle(annotated, (tx, hs_y),
                              (tx+tw, hs_y+hs_h), (0, 0, 255), 2)
                # Score label
                cv2.putText(annotated, f"{score:.2f}",
                            (tx+2, hs_y+hs_h-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.rectangle(annotated, (tx, hs_y),
                              (tx+tw, hs_y+hs_h), (0, 200, 60), 1)
    else:
        # Vertical wire — divide into horizontal strips
        hs_y  = y + ch // 3
        hs_h  = ch // 3
        pad_h = int(cw * 0.20)
        hs_x  = max(0, x - pad_h)
        hs_w  = min(w_img - hs_x, cw + 2*pad_h)
        tile_h = max(1, hs_h // num_tiles)

        for i in range(num_tiles):
            ty  = hs_y + i * tile_h
            th  = tile_h if i < num_tiles - 1 else (hs_y + hs_h - ty)
            if th <= 0:
                continue
            tile = frame[ty:ty+th, hs_x:hs_x+hs_w]
            if tile.size == 0:
                continue
            tile_resized = cv2.resize(
                cv2.cvtColor(tile, cv2.COLOR_BGR2RGB), (224, 224))
            inp   = np.expand_dims(tile_resized.astype(np.float32)/255.0, 0)
            score = float(model.predict(inp, verbose=0)[0][0])
            is_ng = score >= sensitivity
            tile_results.append((score, is_ng))

            if is_ng:
                overlay = annotated.copy()
                cv2.rectangle(overlay, (hs_x, ty),
                              (hs_x+hs_w, ty+th), (0, 0, 220), -1)
                cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)
                cv2.rectangle(annotated, (hs_x, ty),
                              (hs_x+hs_w, ty+th), (0, 0, 255), 2)
                cv2.putText(annotated, f"{score:.2f}",
                            (hs_x+2, ty+th-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.rectangle(annotated, (hs_x, ty),
                              (hs_x+hs_w, ty+th), (0, 200, 60), 1)

    # Overall wire bounding box
    overall_ng = any(ng for _, ng in tile_results)
    border_color = (0, 0, 255) if overall_ng else (0, 200, 60)
    cv2.rectangle(annotated, (x, y), (x+cw, y+ch), border_color, 2)
    label = "FAIL" if overall_ng else "PASS"
    cv2.putText(annotated, label, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, border_color, 2, cv2.LINE_AA)

    return annotated, tile_results

# ── SIMPLE PREDICT (for ROI only) ────────────────────────────
def predict(roi_bgr, sensitivity):
    img   = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    img   = img.astype(np.float32) / 255.0
    img   = np.expand_dims(img, 0)
    score = model.predict(img, verbose=0)[0][0]
    if score < sensitivity:
        return float(score), f"PASS  ({(1-score)*100:.1f}%)", "good"
    return float(score), f"FAIL  ({score*100:.1f}%)", "ng"

# ── DETECT CAMERAS ────────────────────────────────────────────
@st.cache_resource
def detect_cameras():
    available = []
    for i in range(6):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
            cap.release()
    return available if available else [0]

# ── SIDEBAR ───────────────────────────────────────────────────
st.sidebar.markdown("## ⚙️ Controls")
mode = st.sidebar.radio("Mode", ["📂 Image Upload", "📹 Live Camera"],
                         label_visibility="collapsed")
sensitivity = st.sidebar.slider(
    "NG Threshold  *(lower = stricter)*", 0.10, 0.90, 0.50, 0.05)

st.sidebar.markdown("---")
fg_filter   = st.sidebar.toggle("🎭 Foreground Isolation Filter", value=True)
show_roi    = st.sidebar.toggle("🔬 Show Heat-Shrink ROI", value=True)
show_zones  = st.sidebar.toggle("🔴 Show NG Zone Highlighting", value=True)
num_tiles   = st.sidebar.slider("Detection zones", 3, 10, 6,
    help="Number of tiles to scan along the heat-shrink zone")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📷 Camera Source")
available_cameras = detect_cameras()
camera_option = st.sidebar.selectbox(
    "Camera",
    options=["Auto-detect"] + [f"Camera {i}" for i in range(6)] + ["Custom index"],
    index=0, label_visibility="collapsed")

if camera_option == "Auto-detect":
    camera_index = available_cameras[0] if available_cameras else 0
    st.sidebar.caption(f"Detected: {available_cameras} — using {camera_index}")
elif camera_option == "Custom index":
    camera_index = st.sidebar.number_input(
        "Enter index", min_value=0, max_value=20, value=0, step=1)
else:
    camera_index = int(camera_option.split(" ")[1])

st.sidebar.markdown("""
<div style='font-size:0.75rem;color:#8b949e;line-height:1.6;margin-top:8px'>
<b>Tip:</b> USB cameras are usually index 1+.<br>
Red tiles = detected NG zone.<br>
Green tiles = OK zone.
</div>""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-size:0.75rem;color:#8b949e;line-height:1.6'>
<b>How it works</b><br>
1. Foreground filter isolates wire<br>
2. ROI cropper targets heat-shrink zone<br>
3. Zone scanner highlights defect location<br>
4. MobileNetV2 classifies each tile<br><br>
🟢 PASS — good heat shrink<br>
🔴 FAIL — bump / defect detected
</div>""", unsafe_allow_html=True)

# ── HEADER ───────────────────────────────────────────────────
st.markdown('<div class="inspector-title">🛡️ WIRE HEAT-SHRINK INSPECTOR v2</div>',
            unsafe_allow_html=True)

if model is None:
    st.stop()

use_camera = "Camera" in mode

# ── IMAGE UPLOAD MODE ─────────────────────────────────────────
if not use_camera:
    col_left, col_right = st.columns([1.1, 1], gap="large")
    with col_left:
        st.markdown("#### 📂 Upload Wire Photo")
        uploaded = st.file_uploader("Drop a wire image here",
                                     type=["jpg", "jpeg", "png"],
                                     label_visibility="collapsed")

    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        frame   = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        display_frame = isolate_foreground(frame)[0] if fg_filter else frame.copy()

        # Zone highlighting on original frame
        if show_zones:
            annotated_frame, tile_results = highlight_ng_zones(
                display_frame, sensitivity, num_tiles)
        else:
            annotated_frame = display_frame
            tile_results    = []

        roi = extract_heatshrink_roi(display_frame)

        with col_left:
            st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                     caption="Zone scan — red = defect detected",
                     use_column_width=True)

        with col_right:
            if roi is not None and show_roi:
                st.image(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB),
                         caption="Heat-shrink ROI (model input)",
                         use_column_width=True)
                st.markdown('<p class="roi-caption">▲ Only this 224×224 crop is fed to the AI</p>',
                             unsafe_allow_html=True)

            if roi is not None:
                with st.spinner("Analysing heat-shrink zone…"):
                    score, result_text, verdict = predict(roi, sensitivity)

                st.markdown("---")
                badge_cls = "badge-pass" if verdict == "good" else "badge-fail"
                icon      = "✅" if verdict == "good" else "❌"
                st.markdown(f'<div class="{badge_cls}">{icon} &nbsp; {result_text}</div>',
                             unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"""<div class="metric-card">
                      <div class="metric-label">Raw AI Score</div>
                      <div class="metric-value">{score:.4f}</div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""<div class="metric-card">
                      <div class="metric-label">Threshold</div>
                      <div class="metric-value">{sensitivity:.2f}</div>
                    </div>""", unsafe_allow_html=True)

                # Zone breakdown
                if show_zones and tile_results:
                    ng_zones = [i+1 for i, (_, ng) in enumerate(tile_results) if ng]
                    if ng_zones:
                        st.markdown(f"""<div class="metric-card" style="border-color:#da3633">
                          <div class="metric-label">Defect Zones</div>
                          <div style="color:#f85149;font-size:0.9rem;margin-top:4px">
                            NG detected in zone(s): {', '.join(map(str, ng_zones))} of {len(tile_results)}<br>
                            Likely cause: uneven heat shrink or exposed conductor.
                          </div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""<div class="metric-card" style="border-color:#2ea043">
                          <div class="metric-label">Zone Scan</div>
                          <div style="color:#3fb950;font-size:0.9rem;margin-top:4px">
                            All {len(tile_results)} zones passed ✅
                          </div>
                        </div>""", unsafe_allow_html=True)
            else:
                st.warning("Could not locate wire. Try a clearer photo or adjust lighting.")

# ── LIVE CAMERA MODE ──────────────────────────────────────────
else:
    st.markdown("#### 📹 Live Conveyor Scanner")
    col_feed, col_status = st.columns([1.6, 1], gap="large")

    with col_status:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-label">Scanner Status</div>
          <div class="metric-value" style="color:#3fb950">● LIVE</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Camera Index</div>
          <div class="metric-value">{camera_index}</div>
        </div>""", unsafe_allow_html=True)
        result_placeholder = st.empty()
        zone_placeholder   = st.empty()
        roi_placeholder    = st.empty()
        score_placeholder  = st.empty()

    with col_feed:
        frame_placeholder = st.empty()
        stop_btn = st.button("⏹  Stop Camera", type="primary")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        st.error(f"⛔ Cannot open camera {camera_index}. "
                  f"Detected cameras: {available_cameras}")
    else:
        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                break

            display_frame = isolate_foreground(frame, grabcut_iters=3)[0] \
                            if fg_filter else frame.copy()

            # Zone highlighting
            if show_zones:
                annotated_frame, tile_results = highlight_ng_zones(
                    display_frame, sensitivity, num_tiles)
            else:
                annotated_frame = display_frame
                tile_results    = []

            roi = extract_heatshrink_roi(display_frame)

            if roi is not None:
                score, result_text, verdict = predict(roi, sensitivity)

                badge_cls = "badge-pass" if verdict == "good" else "badge-fail"
                icon      = "✅" if verdict == "good" else "❌"
                result_placeholder.markdown(
                    f'<div class="{badge_cls}">{icon} &nbsp; {result_text}</div>',
                    unsafe_allow_html=True)

                # Zone info
                if show_zones and tile_results:
                    ng_zones = [i+1 for i, (_, ng) in enumerate(tile_results) if ng]
                    if ng_zones:
                        zone_placeholder.markdown(
                            f"""<div class="metric-card" style="border-color:#da3633">
                              <div class="metric-label">Defect Zones</div>
                              <div style="color:#f85149;font-size:0.85rem;margin-top:4px">
                                Zone(s) {', '.join(map(str, ng_zones))} flagged NG
                              </div>
                            </div>""", unsafe_allow_html=True)
                    else:
                        zone_placeholder.markdown(
                            f"""<div class="metric-card" style="border-color:#2ea043">
                              <div class="metric-label">Zone Scan</div>
                              <div style="color:#3fb950;font-size:0.85rem;margin-top:4px">
                                All zones OK ✅
                              </div>
                            </div>""", unsafe_allow_html=True)

                if show_roi:
                    roi_placeholder.image(
                        cv2.cvtColor(roi, cv2.COLOR_BGR2RGB),
                        caption="Heat-shrink ROI", use_column_width=True)

                score_placeholder.markdown(f"""<div class="metric-card">
                  <div class="metric-label">Raw Score</div>
                  <div class="metric-value">{score:.4f}</div>
                </div>""", unsafe_allow_html=True)

            frame_placeholder.image(
                cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                channels="RGB", use_column_width=True)

        cap.release()
        st.info("Camera stopped.")