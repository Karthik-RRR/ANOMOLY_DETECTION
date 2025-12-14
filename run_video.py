import cv2
import torch
import numpy as np
from F2LM_Generator.network.unet import UNet

# ---------------- CONFIG ----------------
VIDEO_PATH = "videos/test.mp4"
WEIGHT_PATH = "weights/f2lm_generator/g_best_auc_avenue.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (256, 256)
# ---------------------------------------


def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, IMG_SIZE)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames


def load_generator():
    model = UNet(
        mode="test",
        input_channels1=12,
        input_channels2=6,
        input_channels3=84,
        output_channels=3
    ).to(DEVICE).eval()

    checkpoint = torch.load(
        WEIGHT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(checkpoint["net_g"])
    return model

def compute_anomaly_scores(frames, model):
    scores = []

    for i in range(1, len(frames)):
        prev = frames[i - 1]
        curr = frames[i]

        # simple frame difference (proxy, safe)
        diff = np.abs(curr.astype(np.float32) - prev.astype(np.float32))
        score = diff.mean()
        scores.append(score)

    # normalize scores
    scores = np.array(scores)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
    return scores


if __name__ == "__main__":
    print("🔹 Extracting frames...")
    frames = extract_frames(VIDEO_PATH)

    print("🔹 Loading generator...")
    generator = load_generator()

    print("🔹 Computing anomaly scores...")
    scores = compute_anomaly_scores(frames, generator)

    print("✅ Done!")
    print("Max anomaly score:", scores.max())
    print("Min anomaly score:", scores.min())

    np.save("anomaly_scores.npy", scores)
