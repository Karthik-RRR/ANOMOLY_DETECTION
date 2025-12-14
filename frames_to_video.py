import cv2
import os

# CHANGE THIS PATH
FRAME_DIR = "avenue/testing/frames/02"
OUTPUT_VIDEO = "avenue_test_02.mp4"

FPS = 25
IMG_SIZE = (256, 256)

frames = sorted(os.listdir(FRAME_DIR))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, IMG_SIZE)

for frame_name in frames:
    frame_path = os.path.join(FRAME_DIR, frame_name)
    img = cv2.imread(frame_path)
    if img is None:
        continue

    img = cv2.resize(img, IMG_SIZE)
    out.write(img)

out.release()
print("✅ Video saved as", OUTPUT_VIDEO)
