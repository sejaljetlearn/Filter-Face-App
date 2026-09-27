"""
app.py — Flask backend for the Face Filter Web App
Works both locally AND on cloud (Render, Railway, etc.)

Architecture:
- Browser captures webcam via getUserMedia()
- JS sends frames as base64 to POST /process_frame
- Server processes with OpenCV + selected filter
- Returns processed frame as base64 JPEG
- Also supports local MJPEG mode as fallback
"""

import cv2
import numpy as np
import threading
import time
import base64
import os
import io
from flask import Flask, Response, render_template, request, jsonify, send_file
from filters import apply_filter

app = Flask(__name__)

# ──────────────────────────────────────────────
# Shared state
# ──────────────────────────────────────────────
current_filter = "none"
face_count = 0
last_processed_frame = None  # For capture endpoint
state_lock = threading.Lock()

# Load Haar cascade classifiers once at startup
cascade_path = cv2.data.haarcascades
face_cascade = cv2.CascadeClassifier(
    os.path.join(cascade_path, "haarcascade_frontalface_default.xml")
)
eye_cascade = cv2.CascadeClassifier(
    os.path.join(cascade_path, "haarcascade_eye.xml")
)


def process_frame(frame):
    """Detect faces and apply the current filter."""
    global face_count, last_processed_frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray_eq,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    eyes_list = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20)
        )
        for (ex, ey, ew, eh) in eyes:
            eyes_list.append((x + ex, y + ey, ew, eh))

    with state_lock:
        face_count = len(faces)

    filtered = apply_filter(frame, faces, eyes_list, current_filter)

    # Overlay face count badge
    badge = f"Faces: {face_count}"
    cv2.putText(filtered, badge, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
    cv2.putText(filtered, badge, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 255, 150), 2)

    # Filter name overlay
    cv2.putText(filtered, f"Filter: {current_filter.upper()}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
    cv2.putText(filtered, f"Filter: {current_filter.upper()}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 255), 1)

    with state_lock:
        last_processed_frame = filtered.copy()

    return filtered


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process_frame", methods=["POST"])
def process_frame_route():
    """
    Receive a base64 JPEG frame from the browser,
    process it with OpenCV, return processed base64 JPEG.
    """
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image data"}), 400

    try:
        # Decode base64 image (strip "data:image/...;base64," prefix)
        img_data = data["image"]
        if "," in img_data:
            img_data = img_data.split(",", 1)[1]

        img_bytes = base64.b64decode(img_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Could not decode image"}), 400

        # Process with OpenCV
        result = process_frame(frame)

        # Encode back to JPEG
        _, jpeg = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 80])
        result_b64 = base64.b64encode(jpeg.tobytes()).decode("utf-8")

        return jsonify({
            "image": f"data:image/jpeg;base64,{result_b64}",
            "faces": face_count
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/set_filter", methods=["POST"])
def set_filter():
    global current_filter
    data = request.get_json()
    filter_name = data.get("filter", "none")
    valid = ["none", "dog", "sunglasses", "crown", "rainbow",
             "neon", "vintage", "cyberpunk", "mirror", "sketch", "pixelate",
             "glitch", "thermal", "nightvision", "cartoon", "fisheye", "disco"]
    if filter_name in valid:
        current_filter = filter_name
        return jsonify({"status": "ok", "filter": filter_name})
    return jsonify({"status": "error", "message": "Invalid filter"}), 400


@app.route("/capture", methods=["GET"])
def capture():
    """Return last processed frame as downloadable PNG."""
    with state_lock:
        frame = last_processed_frame
    if frame is None:
        return jsonify({"error": "No frame available"}), 503
    _, png = cv2.imencode(".png", frame)
    ts = time.strftime("%Y%m%d_%H%M%S")
    return send_file(
        io.BytesIO(png.tobytes()),
        mimetype="image/png",
        as_attachment=True,
        download_name=f"face_filter_{ts}.png"
    )


@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "faces": face_count,
        "filter": current_filter,
        "camera_running": True  # Browser handles camera now
    })


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🎭 Face Filter App running at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
