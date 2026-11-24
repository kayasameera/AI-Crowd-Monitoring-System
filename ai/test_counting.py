from ultralytics import YOLO
import cv2

from tracker import track_people
from counting import count_entry_exit, LINE_Y


model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)

previous_positions = {}

entry_count = 0
exit_count = 0


while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera not found")
        break

    # Tracking
    results = track_people(model, frame)

    # Counting
    previous_positions, entry_count, exit_count = count_entry_exit(
        results,
        previous_positions,
        entry_count,
        exit_count
    )

    # Draw counting line
    cv2.line(
        frame,
        (0, LINE_Y),
        (frame.shape[1], LINE_Y),
        (255, 0, 0),
        2
    )

    # Draw people
    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()

        for box, person_id in zip(boxes, ids):

            x1, y1, x2, y2 = map(int, box)

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"ID: {int(person_id)}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )

    # Display counts
    cv2.putText(
        frame,
        f"ENTRY: {entry_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"EXIT: {exit_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.imshow(
        "Entry Exit Counting Test",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
