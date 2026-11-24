from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Heatmap accumulation
heatmap = None

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera not found")
        break

    # Create heatmap array
    if heatmap is None:
        heatmap = np.zeros(
            (frame.shape[0], frame.shape[1]),
            dtype=np.float32
        )

    # Slowly fade old heatmap information
    heatmap *= 0.97

    # YOLO tracking
    results = model.track(
        frame,
        persist=True,
        classes=[0],
        tracker="bytetrack.yaml",
        verbose=False
    )

    # Detect people
    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()

        for box, person_id in zip(boxes, ids):

            x1, y1, x2, y2 = map(int, box)

            # Calculate person center
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Add person's position to heatmap
            cv2.circle(
                heatmap,
                (center_x, center_y),
                50,
                1,
                -1
            )

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Display ID
            cv2.putText(
                frame,
                f"ID: {int(person_id)}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Center point
            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )

    # Blur heatmap
    blurred_heatmap = cv2.GaussianBlur(
        heatmap,
        (0, 0),
        sigmaX=25
    )

    # Normalize heatmap
    normalized_heatmap = cv2.normalize(
        blurred_heatmap,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    normalized_heatmap = normalized_heatmap.astype(np.uint8)

    # Apply heatmap colors
    colored_heatmap = cv2.applyColorMap(
        normalized_heatmap,
        cv2.COLORMAP_JET
    )

    # Create mask so heatmap appears ONLY where activity exists
    mask = normalized_heatmap > 15

    # Blend only heatmap regions
    output = frame.copy()

    output[mask] = cv2.addWeighted(
        frame[mask],
        0.65,
        colored_heatmap[mask],
        0.35,
        0
    )

    # Display
    cv2.imshow(
        "AI Crowd Monitoring - Heatmap",
        output
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()