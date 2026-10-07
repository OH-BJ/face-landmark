"""MediaPipe Face Landmarker - webcam demo.

Keys: q / ESC = quit, m = toggle mesh, b = toggle blendshape panel
"""
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarksConnections as FLC

MODEL_PATH = Path(__file__).with_name("face_landmarker.task")
CAMERA_INDEX = 0
NUM_FACES = 2

latest_result = None


def on_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result


def draw_connections(img, pts, connections, color, thickness=1):
    for c in connections:
        cv2.line(img, pts[c.start], pts[c.end], color, thickness, cv2.LINE_AA)


def draw_face(img, landmarks, show_mesh):
    h, w = img.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    if show_mesh:
        draw_connections(img, pts, FLC.FACE_LANDMARKS_TESSELATION, (80, 80, 80))
    draw_connections(img, pts, FLC.FACE_LANDMARKS_FACE_OVAL, (224, 224, 224), 2)
    draw_connections(img, pts, FLC.FACE_LANDMARKS_LIPS, (80, 80, 255), 2)
    for conns in (FLC.FACE_LANDMARKS_LEFT_EYE, FLC.FACE_LANDMARKS_LEFT_EYEBROW):
        draw_connections(img, pts, conns, (80, 255, 80), 2)
    for conns in (FLC.FACE_LANDMARKS_RIGHT_EYE, FLC.FACE_LANDMARKS_RIGHT_EYEBROW):
        draw_connections(img, pts, conns, (255, 160, 60), 2)
    for conns in (FLC.FACE_LANDMARKS_LEFT_IRIS, FLC.FACE_LANDMARKS_RIGHT_IRIS):
        draw_connections(img, pts, conns, (0, 255, 255), 2)


def draw_blendshapes(img, blendshapes, top_k=10):
    top = sorted(blendshapes, key=lambda c: c.score, reverse=True)[:top_k]
    x0, y0 = 10, 60
    for i, c in enumerate(top):
        y = y0 + i * 20
        cv2.rectangle(img, (x0 + 150, y - 12), (x0 + 150 + int(c.score * 150), y + 2), (0, 200, 255), -1)
        cv2.putText(img, f"{c.category_name[:18]}", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_faces=NUM_FACES,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        result_callback=on_result,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam {CAMERA_INDEX}")

    show_mesh, show_blend = True, True
    prev_t, fps = time.time(), 0.0
    start = time.monotonic()
    last_ts = -1

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror view

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int((time.monotonic() - start) * 1000)
            if ts <= last_ts:  # timestamps must be strictly increasing
                ts = last_ts + 1
            last_ts = ts
            landmarker.detect_async(mp_image, ts)

            result = latest_result
            n_faces = 0
            if result is not None:
                n_faces = len(result.face_landmarks)
                for landmarks in result.face_landmarks:
                    draw_face(frame, landmarks, show_mesh)
                if show_blend and result.face_blendshapes:
                    draw_blendshapes(frame, result.face_blendshapes[0])

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_t, 1e-6))
            prev_t = now
            cv2.putText(frame, f"FPS {fps:.1f}  faces {n_faces}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("MediaPipe Face Landmarker", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                show_mesh = not show_mesh
            if key == ord("b"):
                show_blend = not show_blend

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
