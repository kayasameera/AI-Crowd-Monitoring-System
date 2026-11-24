from ultralytics import YOLO
import cv2

from tracker import track_people
from counting import count_entry_exit, LINE_Y
from occupancy import update_occupancy


model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)

previous_positions = {}

entry_count = 0
exit_count = 0

previous_entry_count = 0
previous_exit_count = 0

occupancy = 0


while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera not found")
        break

    # =====================================
    # TRACK PEOPLE
    # =====================================

    results = track_people(model, frame)

    # =====================================
    # COUNT ENTRY / EXIT
    # =====================================

    previous_positions, entry_count, exit_count = count_entry_exit(
        results,
        previous_positions,
        entry_count,
        exit_count
    )

    # =====================================
    # UPDATE OCCUPANCY
    # =====================================

    occupancy = update_occupancy(
        previous_entry_count,
        previous_exit_count,
        entry_count,
        exit_count,
        occupancy
    )

    previous_entry_count = entry_count
    previous_exit_count = exit_count

    # =====================================
    # COUNTING LINE
    # =====================================

    cv2.line(
        frame,
        (0, LINE_Y),
        (frame.shape[1], LINE_Y),
        (255, 0, 0),
        2
    )

    # =====================================
    # DRAW PEOPLE
    # =====================================

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()

        for box, person_id in zip(boxes, ids):

            x1, y1, x2, y2 = map(int, box)

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

    # =====================================
    # DISPLAY INFORMATION
    # =====================================

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

    cv2.putText(
        frame,
        f"OCCUPANCY: {occupancy}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.imshow(
        "Occupancy Test",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()