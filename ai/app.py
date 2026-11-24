import os
import cv2
import time
import sqlite3
import threading
import winsound
import numpy as np
from datetime import datetime
from flask import Flask, Response, jsonify, render_template

from ultralytics import YOLO

# ============================================================
# AI CROWD MONITORING SYSTEM
# Two cameras + YOLO person detection + tracking
# Entry/exit without a line
# Heatmap for BOTH cameras
# SQLite storage
# Sound alerts
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "crowd_data.db")
MODEL_PATH = os.path.join(BASE_DIR, "yolo11n.pt")

# ---------------- CAMERA SETTINGS ----------------
LAPTOP_CAMERA = 0

# IP Webcam address shown on your phone:
# IPv4: http://192.168.1.4:8080
# Video stream:
MOBILE_CAMERA = "http://192.168.31.86:8080/video"

# ---------------- AI SETTINGS ----------------
CONFIDENCE = 0.45
IOU = 0.50
TRACKER = "bytetrack.yaml"

# A track must be visible for this many frames before it becomes an entry.
MIN_CONFIRM_FRAMES = 3

# If a tracked person is missing this many frames, count an exit.
MISSING_FRAMES = 45

# Ignore extremely tiny detections (helps reduce hand/noise false positives).
MIN_BOX_AREA = 1200

# ---------------- CROWD SETTINGS ----------------
CROWD_LIMIT = 6
LOCAL_CROWD_LIMIT = 4

GRID_ROWS = 3
GRID_COLS = 3

# Save DB metrics at most once per second per camera.
DB_SAVE_INTERVAL = 1.0

# Mobile camera reconnect interval.
RECONNECT_SECONDS = 3

# Sound alert cooldown.
ALERT_COOLDOWN = 10

# Heatmap decay per processed frame.
HEATMAP_DECAY = 0.96


# ============================================================
# DATABASE
# ============================================================

def init_database():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS crowd_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            people_in_frame INTEGER DEFAULT 0,
            entry_count INTEGER DEFAULT 0,
            exit_count INTEGER DEFAULT 0,
            occupancy INTEGER DEFAULT 0,
            density TEXT DEFAULT 'LOW',
            alert_status TEXT DEFAULT 'NORMAL'
        )
    """)

    # Add camera column to an older database if it does not exist.
    cur.execute("PRAGMA table_info(crowd_data)")
    columns = {row[1] for row in cur.fetchall()}

    if "camera" not in columns:
        cur.execute("ALTER TABLE crowd_data ADD COLUMN camera TEXT DEFAULT 'LAPTOP'")

    conn.commit()
    conn.close()


def save_metrics(camera_name, people, entry, exit_count, occupancy, density, status):
    try:
        conn = sqlite3.connect(DB_PATH, timeout=5)
        conn.execute("""
            INSERT INTO crowd_data
            (timestamp, people_in_frame, entry_count, exit_count,
             occupancy, density, alert_status, camera)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            int(people),
            int(entry),
            int(exit_count),
            int(occupancy),
            str(density),
            str(status),
            str(camera_name),
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB] Save error for {camera_name}: {e}")


def read_recent_data(limit=30):
    try:
        conn = sqlite3.connect(DB_PATH, timeout=5)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT timestamp, people_in_frame, entry_count, exit_count,
                   occupancy, density, alert_status, camera
            FROM crowd_data
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        print("[DB] Read error:", e)
        return []


# ============================================================
# SOUND ALERT
# ============================================================

_sound_lock = threading.Lock()
_last_global_beep = 0.0


def play_alert(camera_name):
    global _last_global_beep

    now = time.time()

    with _sound_lock:
        if now - _last_global_beep < ALERT_COOLDOWN:
            return

        _last_global_beep = now

    def beep():
        try:
            print(f"[ALERT] HIGH CROWD - {camera_name}")
            for _ in range(3):
                winsound.Beep(1200, 350)
                time.sleep(0.12)
        except Exception as e:
            print("[ALERT] Sound error:", e)

    threading.Thread(target=beep, daemon=True).start()


# ============================================================
# CAMERA PROCESSOR
# ============================================================

class CameraProcessor:
    def __init__(self, camera_id, name, source):
        self.camera_id = camera_id
        self.name = name
        self.source = source

        self.model = None
        self.cap = None
        self.thread = None
        self.running = False

        self.lock = threading.Lock()

        # Latest processed JPEG.
        self.latest_frame = None

        # Latest heatmap-only JPEG.
        self.latest_heatmap = None

        # Camera state.
        self.connected = False
        self.error_message = "Starting camera..."

        # Tracking state.
        self.track_state = {}
        self.confirm_counts = {}
        self.known_ids = set()
        self.recently_lost = []

        # Metrics.
        self.people = 0
        self.entry_count = 0
        self.exit_count = 0
        self.occupancy = 0
        self.density = "LOW"
        self.alert_status = "NORMAL"
        self.local_counts = [[0 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]

        # Heatmap state.
        self.heatmap = None
        self.frame_width = 640
        self.frame_height = 480

        self.last_db_save = 0.0
        self.last_alert_time = 0.0

    # --------------------------------------------------------
    # Start
    # --------------------------------------------------------
    def start(self):
        self.running = True
        self.thread = threading.Thread(
            target=self.process_loop,
            name=f"camera-{self.camera_id}",
            daemon=True
        )
        self.thread.start()

    # --------------------------------------------------------
    # Open camera
    # --------------------------------------------------------
    def open_camera(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        self.cap = None

        if isinstance(self.source, int):
            print(f"[Camera {self.camera_id}] Opening {self.name}...")
            cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
            cap.set(cv2.CAP_PROP_FPS, 20)
        else:
            print(f"[Camera {self.camera_id}] Opening {self.name}...")
            # FFmpeg backend works better for IP Webcam on Windows.
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if cap.isOpened():
            self.cap = cap
            self.connected = True
            self.error_message = ""
            print(f"[Camera {self.camera_id}] Camera opened successfully")
            return True

        self.connected = False
        self.error_message = "Camera failed to open"
        print(f"[Camera {self.camera_id}] FAILED TO OPEN")

        try:
            cap.release()
        except Exception:
            pass

        return False

    # --------------------------------------------------------
    # Heatmap initialization
    # --------------------------------------------------------
    def init_heatmap(self, frame):
        h, w = frame.shape[:2]
        self.frame_height = h
        self.frame_width = w

        if (
            self.heatmap is None
            or self.heatmap.shape != (h, w)
        ):
            self.heatmap = np.zeros((h, w), dtype=np.float32)

    # --------------------------------------------------------
    # Heatmap update
    # --------------------------------------------------------
    def update_heatmap(self, centers):
        if self.heatmap is None:
            return

        self.heatmap *= HEATMAP_DECAY

        h, w = self.heatmap.shape

        for cx, cy in centers:
            cx = max(0, min(w - 1, int(cx)))
            cy = max(0, min(h - 1, int(cy)))

            # Add a soft circular hotspot.
            radius = 28

            y1 = max(0, cy - radius)
            y2 = min(h, cy + radius + 1)
            x1 = max(0, cx - radius)
            x2 = min(w, cx + radius + 1)

            yy, xx = np.ogrid[y1:y2, x1:x2]
            distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

            blob = np.clip(1.0 - distance / radius, 0, 1) * 4.0
            self.heatmap[y1:y2, x1:x2] += blob.astype(np.float32)

    # --------------------------------------------------------
    # Make heatmap image
    # --------------------------------------------------------
    def make_heatmap_image(self, frame):
        if self.heatmap is None:
            return frame.copy()

        heat = cv2.GaussianBlur(self.heatmap, (0, 0), 18)

        max_value = float(np.max(heat))

        if max_value <= 0.01:
            return np.zeros_like(frame)

        normalized = np.clip((heat / max_value) * 255.0, 0, 255).astype(np.uint8)

        color = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

        # Dark background with heat areas visible.
        result = cv2.addWeighted(
            np.zeros_like(frame),
            0.35,
            color,
            0.85,
            0
        )

        return result

    # --------------------------------------------------------
    # Draw camera information
    # --------------------------------------------------------
    def draw_overlay(self, frame):
        status_color = (0, 0, 255) if self.alert_status == "ALERT" else (0, 220, 0)

        # Main title bar.
        cv2.rectangle(frame, (0, 0), (470, 105), (20, 20, 20), -1)

        cv2.putText(
            frame,
            "AI CROWD MONITORING SYSTEM",
            (12, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"CAMERA: {self.name}",
            (12, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 220, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"PEOPLE: {self.people}",
            (12, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"ENTRY: {self.entry_count}",
            (165, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"EXIT: {self.exit_count}",
            (305, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"DENSITY: {self.density}",
            (12, 97),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"STATUS: {self.alert_status}",
            (205, 97),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            status_color,
            2,
            cv2.LINE_AA
        )

        return frame

    # --------------------------------------------------------
    # Process YOLO frame
    # --------------------------------------------------------
    def process_frame(self, frame):
        self.init_heatmap(frame)

        # YOLO tracking.
        results = self.model.track(
            frame,
            persist=True,
            classes=[0],          # COCO person class
            conf=CONFIDENCE,
            iou=IOU,
            tracker=TRACKER,
            verbose=False
        )

        result = results[0]

        boxes = []
        ids = []
        centers = []

        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()

            if result.boxes.conf is not None:
                confs = result.boxes.conf.cpu().numpy()
            else:
                confs = np.ones(len(xyxy))

            if result.boxes.id is not None:
                track_ids = result.boxes.id.int().cpu().tolist()
            else:
                track_ids = [None] * len(xyxy)

            for box, conf, tid in zip(xyxy, confs, track_ids):
                x1, y1, x2, y2 = map(int, box)

                area = max(0, x2 - x1) * max(0, y2 - y1)

                if area < MIN_BOX_AREA:
                    continue

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                boxes.append((x1, y1, x2, y2, float(conf), tid))
                centers.append((cx, cy))

        # Update heatmap using detected person centers.
        self.update_heatmap(centers)

        # Current frame people.
        self.people = len(boxes)
        self.occupancy = self.people

        # Grid/local crowd calculation.
        self.local_counts = [
            [0 for _ in range(GRID_COLS)]
            for _ in range(GRID_ROWS)
        ]

        h, w = frame.shape[:2]

        for cx, cy in centers:
            col = min(GRID_COLS - 1, max(0, int(cx / max(w, 1) * GRID_COLS)))
            row = min(GRID_ROWS - 1, max(0, int(cy / max(h, 1) * GRID_ROWS)))
            self.local_counts[row][col] += 1

        max_local = max(
            max(row) for row in self.local_counts
        ) if self.local_counts else 0

        # Density.
        if self.people >= CROWD_LIMIT or max_local >= LOCAL_CROWD_LIMIT:
            self.density = "HIGH"
            self.alert_status = "ALERT"
        elif self.people >= max(2, CROWD_LIMIT // 2):
            self.density = "MEDIUM"
            self.alert_status = "NORMAL"
        else:
            self.density = "LOW"
            self.alert_status = "NORMAL"

        # Tracking / entry / exit.
        current_ids = set()

        for x1, y1, x2, y2, conf, tid in boxes:
            if tid is None:
                continue

            tid = int(tid)
            current_ids.add(tid)

            if tid not in self.track_state:
                self.track_state[tid] = {
                    "missing": 0,
                    "frames": 0,
                    "last_center": ((x1 + x2) // 2, (y1 + y2) // 2),
                    "last_seen": time.time(),
                    "active": False,
                }

            state = self.track_state[tid]
            state["missing"] = 0
            state["frames"] += 1
            state["last_seen"] = time.time()
            state["last_center"] = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Count only after a stable track.
            if (
                not state["active"]
                and state["frames"] >= MIN_CONFIRM_FRAMES
            ):
                state["active"] = True

                if tid not in self.known_ids:
                    self.entry_count += 1
                    self.known_ids.add(tid)

                    print(
                        f"[Camera {self.camera_id}] "
                        f"ENTRY ID={tid} Total={self.entry_count}"
                    )

        # Missing-track handling.
        for tid in list(self.track_state.keys()):
            if tid not in current_ids:
                self.track_state[tid]["missing"] += 1

                if self.track_state[tid]["missing"] >= MISSING_FRAMES:
                    state = self.track_state[tid]

                    if state["active"]:
                        self.exit_count += 1

                        self.recently_lost.append({
                            "time": time.time(),
                            "center": state["last_center"],
                            "id": tid
                        })

                        print(
                            f"[Camera {self.camera_id}] "
                            f"EXIT ID={tid} Total={self.exit_count}"
                        )

                    del self.track_state[tid]

        # Remove old recently lost entries.
        cutoff = time.time() - 5
        self.recently_lost = [
            item for item in self.recently_lost
            if item["time"] >= cutoff
        ]

        # Draw heatmap over original frame.
        heat_overlay = self.make_heatmap_image(frame)

        # Only blend strongly where heat exists.
        if self.heatmap is not None and np.max(self.heatmap) > 0.01:
            heat_norm = np.clip(
                self.heatmap / max(float(np.max(self.heatmap)), 0.01),
                0,
                1
            )

            alpha = (heat_norm * 0.48).astype(np.float32)
            alpha3 = np.repeat(alpha[:, :, None], 3, axis=2)

            frame_float = frame.astype(np.float32)
            heat_float = heat_overlay.astype(np.float32)

            frame = (
                frame_float * (1.0 - alpha3)
                + heat_float * alpha3
            ).astype(np.uint8)

        # Draw bounding boxes AFTER heatmap, so boxes remain clear.
        for x1, y1, x2, y2, conf, tid in boxes:
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = "PERSON"
            if tid is not None:
                label += f" ID:{int(tid)}"

            cv2.putText(
                frame,
                label,
                (x1, max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            # Center point used for heatmap.
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            cv2.circle(
                frame,
                (cx, cy),
                4,
                (0, 0, 255),
                -1
            )

        # Draw 3x3 grid lightly to show local crowd zones.
        for c in range(1, GRID_COLS):
            x = int(w * c / GRID_COLS)
            cv2.line(frame, (x, 0), (x, h), (255, 255, 0), 1)

        for r in range(1, GRID_ROWS):
            y = int(h * r / GRID_ROWS)
            cv2.line(frame, (0, y), (w, y), (255, 255, 0), 1)

        frame = self.draw_overlay(frame)

        # Separate heatmap-only image.
        heatmap_display = heat_overlay.copy()

        cv2.putText(
            heatmap_display,
            f"{self.name} - CROWD HEATMAP",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Alert sound.
        if self.alert_status == "ALERT":
            play_alert(self.name)

        # Save metrics.
        now = time.time()

        if now - self.last_db_save >= DB_SAVE_INTERVAL:
            self.last_db_save = now

            save_metrics(
                self.name,
                self.people,
                self.entry_count,
                self.exit_count,
                self.occupancy,
                self.density,
                self.alert_status
            )

        return frame, heatmap_display

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------
    def process_loop(self):
        print(f"[Camera {self.camera_id}] Loading YOLO model...")

        try:
            self.model = YOLO(MODEL_PATH)
            print(f"[Camera {self.camera_id}] YOLO model loaded successfully")
        except Exception as e:
            self.error_message = f"YOLO load error: {e}"
            print(f"[Camera {self.camera_id}] YOLO ERROR:", e)
            return

        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self.connected = False

                if not self.open_camera():
                    time.sleep(RECONNECT_SECONDS)
                    continue

            ok, frame = self.cap.read()

            if not ok or frame is None:
                print(f"[Camera {self.camera_id}] Frame failed")
                self.connected = False

                try:
                    self.cap.release()
                except Exception:
                    pass

                self.cap = None

                # Keep retrying the camera instead of killing the application.
                time.sleep(RECONNECT_SECONDS)
                continue

            try:
                processed, heatmap = self.process_frame(frame)

                ok1, jpg = cv2.imencode(
                    ".jpg",
                    processed,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 82]
                )

                ok2, heat_jpg = cv2.imencode(
                    ".jpg",
                    heatmap,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 82]
                )

                if ok1:
                    with self.lock:
                        self.latest_frame = jpg.tobytes()
                        self.connected = True
                        self.error_message = ""

                if ok2:
                    with self.lock:
                        self.latest_heatmap = heat_jpg.tobytes()

            except Exception as e:
                print(f"[Camera {self.camera_id}] Processing error: {e}")
                self.error_message = str(e)

    # --------------------------------------------------------
    # MJPEG frame generator
    # --------------------------------------------------------
    def mjpeg_generator(self):
        while True:
            with self.lock:
                frame = self.latest_frame

            if frame is None:
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)

                cv2.putText(
                    placeholder,
                    f"WAITING FOR {self.name.upper()}...",
                    (110, 235),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                cv2.putText(
                    placeholder,
                    self.error_message[:70],
                    (25, 280),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (180, 180, 180),
                    1,
                    cv2.LINE_AA
                )

                ok, enc = cv2.imencode(".jpg", placeholder)
                if ok:
                    frame = enc.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame +
                b"\r\n"
            )

            time.sleep(0.03)

    # --------------------------------------------------------
    # Heatmap generator
    # --------------------------------------------------------
    def heatmap_generator(self):
        while True:
            with self.lock:
                frame = self.latest_heatmap

            if frame is None:
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)

                cv2.putText(
                    placeholder,
                    f"WAITING FOR {self.name.upper()} HEATMAP...",
                    (65, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                ok, enc = cv2.imencode(".jpg", placeholder)
                if ok:
                    frame = enc.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame +
                b"\r\n"
            )

            time.sleep(0.08)

    # --------------------------------------------------------
    # JSON state
    # --------------------------------------------------------
    def state(self):
        with self.lock:
            connected = self.connected
            error = self.error_message

        return {
            "id": self.camera_id,
            "name": self.name,
            "connected": connected,
            "error": error,
            "people": int(self.people),
            "entry": int(self.entry_count),
            "exit": int(self.exit_count),
            "occupancy": int(self.occupancy),
            "density": self.density,
            "status": self.alert_status,
            "local_counts": self.local_counts,
        }


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)

init_database()

camera0 = CameraProcessor(
    0,
    "LAPTOP CAMERA",
    LAPTOP_CAMERA
)

camera1 = CameraProcessor(
    1,
    "MOBILE CAMERA",
    MOBILE_CAMERA
)

camera0.start()
camera1.start()


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


# Compatibility route for old dashboard versions.
@app.route("/video_feed")
def old_video_feed():
    return Response(
        camera0.mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/video_feed/<int:camera_id>")
def video_feed(camera_id):
    if camera_id == 0:
        camera = camera0
    elif camera_id == 1:
        camera = camera1
    else:
        return "Invalid camera", 404

    return Response(
        camera.mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/heatmap/<int:camera_id>")
def heatmap_feed(camera_id):
    if camera_id == 0:
        camera = camera0
    elif camera_id == 1:
        camera = camera1
    else:
        return "Invalid camera", 404

    return Response(
        camera.heatmap_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/crowd")
def api_crowd():
    c0 = camera0.state()
    c1 = camera1.state()

    total_people = c0["people"] + c1["people"]
    total_entry = c0["entry"] + c1["entry"]
    total_exit = c0["exit"] + c1["exit"]
    total_occupancy = c0["occupancy"] + c1["occupancy"]

    densities = [c0["density"], c1["density"]]

    if "HIGH" in densities:
        total_density = "HIGH"
    elif "MEDIUM" in densities:
        total_density = "MEDIUM"
    else:
        total_density = "LOW"

    if c0["status"] == "ALERT" or c1["status"] == "ALERT":
        total_status = "ALERT"
    else:
        total_status = "NORMAL"

    return jsonify({
        "people": total_people,
        "entry": total_entry,
        "exit": total_exit,
        "occupancy": total_occupancy,
        "density": total_density,
        "status": total_status,
        "cameras": [c0, c1],
        "recent": read_recent_data(30),
    })


if __name__ == "__main__":
    print("\n" + "=" * 64)
    print(" AI CROWD MONITORING SYSTEM")
    print("=" * 64)
    print("Dashboard:")
    print("http://127.0.0.1:5000/")
    print()
    print("Laptop camera:")
    print("http://127.0.0.1:5000/video_feed/0")
    print()
    print("Mobile camera:")
    print("http://127.0.0.1:5000/video_feed/1")
    print()
    print("Laptop heatmap:")
    print("http://127.0.0.1:5000/heatmap/0")
    print()
    print("Mobile heatmap:")
    print("http://127.0.0.1:5000/heatmap/1")
    print("=" * 64)
    print()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )
