import streamlit as st
import cv2
import numpy as np
import os
from datetime import timedelta
import imageio
import base64
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
IMG_SIZE = (256, 256)
THRESHOLD = 30
MIN_AREA = 300
ANOMALY_SCORE_TH = 0.6
VIDEO_DIR = "streamlit_videos"

os.makedirs(VIDEO_DIR, exist_ok=True)
# ---------------------------------------


# ---------- CORE LOGIC ----------
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, IMG_SIZE)
        frames.append(frame)
    cap.release()
    return frames


def compute_anomaly_scores(frames):
    scores = []
    for i in range(1, len(frames)):
        diff = np.abs(frames[i].astype(np.float32) - frames[i - 1].astype(np.float32))
        scores.append(diff.mean())

    scores = np.array(scores)
    if len(scores) == 0:
        return np.array([0.0])

    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
    return scores


def generate_anomaly_gif(input_path, output_gif, draw_boxes):
    cap = cv2.VideoCapture(input_path)
    frames_gif = []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:
        fps = 10

    ret, prev = cap.read()
    if not ret:
        cap.release()
        return

    prev = cv2.resize(prev, IMG_SIZE)
    frame_idx = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, IMG_SIZE)

        if draw_boxes:
            diff = cv2.absdiff(frame, prev)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, THRESHOLD, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                if cv2.contourArea(cnt) > MIN_AREA:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                    timestamp = str(timedelta(seconds=int(frame_idx / fps)))
                    label = f"ANOMALY | {timestamp}"

                    cv2.putText(
                        frame,
                        label,
                        (x, y - 10 if y > 20 else y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA
                    )

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_gif.append(frame_rgb)

        prev = frame.copy()
        frame_idx += 1

    cap.release()
    imageio.mimsave(output_gif, frames_gif, fps=8)


# ---------- NEW PAGE DISPLAY ----------
def open_gif_new_page(gif_path):
    with open(gif_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    html = f"""
    <html>
      <body style="margin:0;background:black;text-align:center;">
        <img src="data:image/gif;base64,{b64}" style="max-width:100%;">
      </body>
    </html>
    """
    components.html(html, height=650)


# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Video Anomaly Detection", layout="centered")
st.title("🚨 Video Anomaly Detection System")

st.markdown("""
**Model Type:** Unsupervised Video Anomaly Detection  
• Normal video → ✅ No boxes  
• Abnormal video → 🔴 ANOMALY + timestamp
""")

uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

# ---------- INPUT VIDEO ----------
if uploaded:
    input_video = os.path.join(VIDEO_DIR, uploaded.name)

    with open(input_video, "wb") as f:
        f.write(uploaded.read())

    st.session_state["input_video"] = input_video

# Always show input video if available
if "input_video" in st.session_state:
    st.subheader("🎥 Input Video")
    with open(st.session_state["input_video"], "rb") as f:
        st.video(f.read())

# ---------- RUN DETECTION ----------
if "input_video" in st.session_state and st.button("Run Anomaly Detection"):
    with st.spinner("Analyzing video..."):
        frames = extract_frames(st.session_state["input_video"])
        scores = compute_anomaly_scores(frames)

        max_score = float(scores.max())
        is_anomaly = max_score > ANOMALY_SCORE_TH

        output_gif = st.session_state["input_video"].replace(".mp4", "_anomaly.gif")

        generate_anomaly_gif(
            st.session_state["input_video"],
            output_gif,
            draw_boxes=is_anomaly
        )

        st.session_state["scores"] = scores
        st.session_state["output_gif"] = output_gif
        st.session_state["is_anomaly"] = is_anomaly
        st.session_state["max_score"] = max_score

    st.success("Processing complete!")

# ---------- OUTPUT ----------
if "output_gif" in st.session_state:
    st.subheader("📊 Result Summary")
    st.write("**Max anomaly score:**", st.session_state["max_score"])

    if st.session_state["is_anomaly"]:
        st.error("⚠️ ANOMALY DETECTED")
    else:
        st.success("✅ NO ANOMALY DETECTED")

    # 📈 GRAPH
    st.subheader("📈 Anomaly Score Graph")
    fig, ax = plt.subplots()
    ax.plot(st.session_state["scores"], color="red")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Anomaly Score")
    st.pyplot(fig)

    # 🎯 OUTPUT GIF
    st.subheader("🎯 Detection Output")
    with open(st.session_state["output_gif"], "rb") as f:
        st.image(f.read())

    if st.button("🔴 Open Output in New Page"):
        open_gif_new_page(st.session_state["output_gif"])
