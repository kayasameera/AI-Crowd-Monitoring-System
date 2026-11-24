from ultralytics import YOLO
import cv2


# ============================================================
# TRACKER SETTINGS
# ============================================================

# Higher confidence prevents weak/false person detections
CONFIDENCE = 0.65

# Only detect the COCO PERSON class
PERSON_CLASS = 0

# Minimum bounding-box dimensions
MIN_WIDTH = 35
MIN_HEIGHT = 60

# Minimum bounding-box area
MIN_AREA = 2500


# ============================================================
# TRACK PEOPLE
# ============================================================

def track_people(model, frame):

    results = model.track(

        frame,

        persist=True,

        classes=[PERSON_CLASS],

        conf=CONFIDENCE,

        iou=0.50,

        tracker="bytetrack.yaml",

        verbose=False

    )


    # ========================================================
    # FILTER BAD DETECTIONS
    # ========================================================

    if results[0].boxes is None:

        return results


    boxes = results[0].boxes


    if len(boxes) == 0:

        return results


    valid_indices = []


    for i in range(len(boxes)):

        xyxy = boxes.xyxy[i].cpu().numpy()

        x1, y1, x2, y2 = map(int, xyxy)


        width = x2 - x1

        height = y2 - y1

        area = width * height


        # ----------------------------------------------------
        # Reject very small detections
        # ----------------------------------------------------

        if width < MIN_WIDTH:

            continue


        if height < MIN_HEIGHT:

            continue


        if area < MIN_AREA:

            continue


        # ----------------------------------------------------
        # Confidence check
        # ----------------------------------------------------

        confidence = float(
            boxes.conf[i].cpu().numpy()
        )


        if confidence < CONFIDENCE:

            continue


        valid_indices.append(i)


    # ========================================================
    # NO VALID PEOPLE
    # ========================================================

    if len(valid_indices) == 0:

        return []


    # ========================================================
    # KEEP ONLY VALID DETECTIONS
    # ========================================================

    filtered_boxes = boxes[valid_indices]


    results[0].boxes = filtered_boxes


    return results