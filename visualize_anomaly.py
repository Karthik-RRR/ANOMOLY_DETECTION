import cv2
import numpy as np

VIDEO_PATH = "videos/test.mp4"
OUTPUT_PATH = "videos/output_anomaly.mp4"
IMG_SIZE = (256, 256)
THRESHOLD = 30  # adjust if needed


def detect_anomaly_regions(prev_frame, curr_frame):
    diff = cv2.absdiff(curr_frame, prev_frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # amplify differences
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, THRESHOLD, 255, cv2.THRESH_BINARY)

    # find contours (anomaly regions)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    return contours, thresh


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, IMG_SIZE)

    ret, prev = cap.read()
    prev = cv2.resize(prev, IMG_SIZE)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, IMG_SIZE)

        contours, mask = detect_anomaly_regions(prev, frame)

        # draw red boxes
        for cnt in contours:
            if cv2.contourArea(cnt) > 300:  # ignore noise
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        out.write(frame)
        prev = frame.copy()

    cap.release()
    out.release()
    print("✅ Output saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
