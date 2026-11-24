from ultralytics import YOLO
import cv2
import numpy as np
import time

from tracker import track_people
from counting import count_entry_exit
from density import calculate_density
from alerts import check_overcrowding
from database import create_database, save_crowd_data


# ==========================================
# LOAD YOLO11 MODEL
# ==========================================

print("Loading YOLO11 model...")

model = YOLO("yolo11n.pt")

print("YOLO11 model loaded successfully.")


# ==========================================
# OPEN CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Camera not found")
    exit()

print("Camera opened successfully.")


# ==========================================
# VARIABLES
# ==========================================

previous_positions = {}

entry_count = 0
exit_count = 0

occupancy = 0

heatmap = None


# ==========================================
# DATABASE
# ==========================================

create_database()

last_database_save = time.time()


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    # ======================================
    # READ CAMERA FRAME
    # ======================================

    ret, frame = cap.read()

    if not ret:

        print("Failed to read camera frame")

        break


    # ======================================
    # CREATE HEATMAP
    # ======================================

    if heatmap is None:

        heatmap = np.zeros(
            (frame.shape[0], frame.shape[1]),
            dtype=np.float32
        )


    # ======================================
    # PERSON TRACKING
    # ======================================

    results = track_people(
        model,
        frame
    )


    # ======================================
    # ENTRY / EXIT COUNTING
    # ======================================

    (
        previous_positions,
        entry_count,
        exit_count
    ) = count_entry_exit(
        results,
        previous_positions,
        entry_count,
        exit_count
    )


    # ======================================
    # PEOPLE CURRENTLY IN FRAME
    # ======================================

    current_people = 0

    boxes = []
    ids = []

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()

        ids = results[0].boxes.id.cpu().numpy()

        current_people = len(ids)


    # ======================================
    # OCCUPANCY
    # ======================================
    #
    # Occupancy now represents the number
    # of people currently detected.
    #
    # This prevents:
    #
    # ENTRY - EXIT
    #
    # from giving misleading occupancy values.
    #

    occupancy = current_people


    # ======================================
    # PROCESS EACH PERSON
    # ======================================

    for box, person_id in zip(boxes, ids):

        x1, y1, x2, y2 = map(
            int,
            box
        )


        # ==================================
        # CENTER POINT
        # ==================================

        center_x = (x1 + x2) // 2

        center_y = (y1 + y2) // 2


        # ==================================
        # HEATMAP
        # ==================================

        cv2.circle(
            heatmap,
            (center_x, center_y),
            40,
            1,
            -1
        )


        # ==================================
        # BOUNDING BOX
        # ==================================

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # ==================================
        # PERSON ID
        # ==================================

        cv2.putText(
            frame,
            f"ID: {int(person_id)}",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


        # ==================================
        # CENTER POINT
        # ==================================

        cv2.circle(
            frame,
            (center_x, center_y),
            5,
            (0, 0, 255),
            -1
        )


    # ==========================================
    # CROWD DENSITY
    # ==========================================

    density = calculate_density(
        current_people
    )


    # ==========================================
    # OVERCROWDING
    # ==========================================

    overcrowding = check_overcrowding(
        current_people
    )


    # ==========================================
    # ALERT STATUS
    # ==========================================

    if overcrowding:

        alert_status = "OVER_CROWDING"

    else:

        alert_status = "NORMAL"


    # ==========================================
    # COUNTING LINE
    # ==========================================

    

    # ==========================================
    # HEATMAP FADE
    # ==========================================
    #
    # Slowly reduce old heatmap activity.
    #

    heatmap *= 0.97


    # ==========================================
    # HEATMAP PROCESSING
    # ==========================================

    blurred_heatmap = cv2.GaussianBlur(
        heatmap,
        (0, 0),
        sigmaX=25
    )


    normalized_heatmap = cv2.normalize(
        blurred_heatmap,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )


    normalized_heatmap = normalized_heatmap.astype(
        np.uint8
    )


    colored_heatmap = cv2.applyColorMap(
        normalized_heatmap,
        cv2.COLORMAP_JET
    )


    # ==========================================
    # HEATMAP MASK
    # ==========================================
    #
    # Only display heatmap where there is
    # meaningful activity.
    #

    mask = normalized_heatmap > 15


    # ==========================================
    # COMBINE CAMERA + HEATMAP
    # ==========================================

    output = frame.copy()


    if np.any(mask):

        output[mask] = cv2.addWeighted(
            frame[mask],
            0.65,
            colored_heatmap[mask],
            0.35,
            0
        )


    # ==========================================
    # DISPLAY TITLE
    # ==========================================

    cv2.putText(
        output,
        "AI CROWD MONITORING SYSTEM",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # ==========================================
    # DISPLAY ENTRY
    # ==========================================

    cv2.putText(
        output,
        f"ENTRY: {entry_count}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


    # ==========================================
    # DISPLAY EXIT
    # ==========================================

    cv2.putText(
        output,
        f"EXIT: {exit_count}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )


    # ==========================================
    # DISPLAY OCCUPANCY
    # ==========================================

    cv2.putText(
        output,
        f"OCCUPANCY: {occupancy}",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )


    # ==========================================
    # DISPLAY PEOPLE IN FRAME
    # ==========================================

    cv2.putText(
        output,
        f"PEOPLE IN FRAME: {current_people}",
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ==========================================
    # DISPLAY DENSITY
    # ==========================================

    cv2.putText(
        output,
        f"DENSITY: {density}",
        (20, 215),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )


    # ==========================================
    # DISPLAY SYSTEM STATUS
    # ==========================================

    if overcrowding:

        cv2.putText(
            output,
            "WARNING: OVERCROWDING!",
            (20, 255),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            3
        )

    else:

        cv2.putText(
            output,
            "STATUS: NORMAL",
            (20, 255),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    # ==========================================
    # SAVE DATA TO DATABASE
    # ==========================================

    if time.time() - last_database_save >= 5:

        save_crowd_data(
            current_people,
            entry_count,
            exit_count,
            occupancy,
            density,
            alert_status
        )

        last_database_save = time.time()


    # ==========================================
    # SHOW FINAL WINDOW
    # ==========================================

    cv2.imshow(
        "AI Crowd Monitoring System - Final",
        output
    )


    # ==========================================
    # PRESS Q TO EXIT
    # ==========================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==========================================
# RELEASE CAMERA
# ==========================================

cap.release()

cv2.destroyAllWindows()

print("Camera stopped.")
print("AI Crowd Monitoring System closed.")