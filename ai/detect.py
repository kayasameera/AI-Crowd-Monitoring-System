from ultralytics import YOLO
import cv2
import numpy as np

# ==========================================
# LOAD YOLO MODEL
# ==========================================

model = YOLO("yolo11n.pt")

# ==========================================
# OPEN WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

# ==========================================
# VARIABLES
# ==========================================

entry_count = 0
exit_count = 0
occupancy = 0

previous_positions = {}

LINE_Y = 300

# Overcrowding threshold
OVER_CROWD_LIMIT = 7

# Heatmap
heatmap = None

# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera not found")
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
    # YOLO TRACKING
    # ======================================

    results = model.track(
        frame,
        persist=True,
        classes=[0],
        tracker="bytetrack.yaml",
        verbose=False
    )

    current_people = 0

    # ======================================
    # COUNTING LINE
    # ======================================

    cv2.line(
        frame,
        (0, LINE_Y),
        (frame.shape[1], LINE_Y),
        (255, 0, 0),
        2
    )

    # ======================================
    # PERSON DETECTION
    # ======================================

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()

        current_people = len(ids)

        # ==================================
        # PROCESS EACH PERSON
        # ==================================

        for box, person_id in zip(boxes, ids):

            x1, y1, x2, y2 = map(int, box)

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            person_id = int(person_id)

            # ==================================
            # HEATMAP ACCUMULATION
            # ==================================

            cv2.circle(
                heatmap,
                (center_x, center_y),
                30,
                1,
                -1
            )

            # ==================================
            # DRAW BOUNDING BOX
            # ==================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # ==================================
            # DISPLAY PERSON ID
            # ==================================

            cv2.putText(
                frame,
                f"ID: {person_id}",
                (x1, y1 - 10),
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

            # ==================================
            # ENTRY / EXIT
            # ==================================

            if person_id in previous_positions:

                previous_y = previous_positions[person_id]

                # ENTRY
                if previous_y < LINE_Y and center_y >= LINE_Y:

                    entry_count += 1
                    occupancy += 1

                    print(
                        f"Person {person_id} ENTERED | "
                        f"Occupancy: {occupancy}"
                    )

                # EXIT
                elif previous_y > LINE_Y and center_y <= LINE_Y:

                    exit_count += 1
                    occupancy = max(
                        0,
                        occupancy - 1
                    )

                    print(
                        f"Person {person_id} EXITED | "
                        f"Occupancy: {occupancy}"
                    )

            previous_positions[person_id] = center_y

    # ==========================================
    # CROWD DENSITY
    # ==========================================

    if current_people <= 3:

        density = "LOW"

    elif current_people <= 7:

        density = "MEDIUM"

    else:

        density = "HIGH"

    # ==========================================
    # OVERCROWDING
    # ==========================================

    if current_people >= OVER_CROWD_LIMIT:

        overcrowding = True

        status_text = "WARNING: OVERCROWDING!"

        status_color = (0, 0, 255)

        print(
            f"WARNING: OVERCROWDING - "
            f"{current_people} people detected"
        )

    else:

        overcrowding = False

        status_text = "STATUS: NORMAL"

        status_color = (0, 255, 0)

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
    # HEATMAP OVERLAY
    # ==========================================

    output = cv2.addWeighted(
        frame,
        0.65,
        colored_heatmap,
        0.35,
        0
    )

    # ==========================================
    # DISPLAY INFORMATION
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

    cv2.putText(
        output,
        f"ENTRY: {entry_count}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        output,
        f"EXIT: {exit_count}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.putText(
        output,
        f"OCCUPANCY: {occupancy}",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.putText(
        output,
        f"PEOPLE IN FRAME: {current_people}",
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"DENSITY: {density}",
        (20, 215),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        output,
        status_text,
        (20, 255),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2
    )

    # ==========================================
    # DISPLAY FINAL OUTPUT
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
# RELEASE
# ==========================================

cap.release()
cv2.destroyAllWindows()