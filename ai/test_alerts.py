from ultralytics import YOLO
import cv2

from tracker import track_people
from alerts import check_overcrowding


model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera not found")
        break

    # Track people
    results = track_people(model, frame)

    # Count people
    current_people = 0

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()

        current_people = len(ids)

        # Draw people
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

    # Check overcrowding
    overcrowding = check_overcrowding(current_people)

    # Display people count
    cv2.putText(
        frame,
        f"PEOPLE: {current_people}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # Display alert
    if overcrowding:

        cv2.putText(
            frame,
            "WARNING: OVERCROWDING!",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            3
        )

        print(
            f"WARNING: OVERCROWDING - "
            f"{current_people} people detected"
        )

    else:

        cv2.putText(
            frame,
            "STATUS: NORMAL",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "Overcrowding Detection Test",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()