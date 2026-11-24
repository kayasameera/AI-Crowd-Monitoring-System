from ultralytics import YOLO
import cv2

from tracker import track_people
from density import calculate_density


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

    # Calculate density
    density = calculate_density(current_people)

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

    # Display density
    cv2.putText(
        frame,
        f"DENSITY: {density}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "Crowd Density Test",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()