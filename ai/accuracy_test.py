import cv2
import csv
import os
import time
import statistics

from ultralytics import YOLO

from tracker import track_people


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_INDEX = 0

# Number of frames used for each test
FRAMES_PER_TEST = 100

# Capture approximately every N seconds
SAMPLE_INTERVAL = 0.10

# Model
MODEL_FILES = [
    "yolo11s.pt",
    "yolo11n.pt"
]

# Confidence used only for displaying fallback detection
CONFIDENCE = 0.65

# CSV output
CSV_FILE = "accuracy_results.csv"


# ============================================================
# COLORS
# ============================================================

GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    model_path = None

    for filename in MODEL_FILES:

        if os.path.exists(filename):

            model_path = filename
            break

    if model_path is None:

        print()
        print("No YOLO model found.")
        print()
        print("Expected one of:")
        print("  yolo11s.pt")
        print("  yolo11n.pt")
        print()

        print("Ultralytics will try to download yolo11s.pt.")

        model_path = "yolo11s.pt"

    print()
    print("Loading YOLO model...")
    print(f"Model: {model_path}")

    model = YOLO(model_path)

    print("YOLO model loaded successfully.")

    return model


# ============================================================
# GET DETECTIONS
# ============================================================

def get_detections(model, frame):

    try:

        results = track_people(
            model,
            frame
        )

    except Exception as error:

        print()
        print("Tracker error:")
        print(error)
        return []


    # --------------------------------------------------------
    # tracker.py may return an empty list
    # --------------------------------------------------------

    if results is None:

        return []


    if len(results) == 0:

        return []


    result = results[0]


    if result.boxes is None:

        return []


    if len(result.boxes) == 0:

        return []


    detections = []


    boxes = result.boxes.xyxy.cpu().numpy()


    # --------------------------------------------------------
    # IDs
    # --------------------------------------------------------

    if result.boxes.id is not None:

        ids = result.boxes.id.cpu().numpy()

    else:

        ids = [-1] * len(boxes)


    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    if result.boxes.conf is not None:

        confidences = result.boxes.conf.cpu().numpy()

    else:

        confidences = [0.0] * len(boxes)


    # --------------------------------------------------------
    # Classes
    # --------------------------------------------------------

    if result.boxes.cls is not None:

        classes = result.boxes.cls.cpu().numpy()

    else:

        classes = [0] * len(boxes)


    # --------------------------------------------------------
    # Process detections
    # --------------------------------------------------------

    for box, person_id, confidence, class_id in zip(

        boxes,
        ids,
        confidences,
        classes

    ):

        # Only person class

        if int(class_id) != 0:

            continue


        x1, y1, x2, y2 = map(
            int,
            box
        )


        detections.append({

            "box": (
                x1,
                y1,
                x2,
                y2
            ),

            "id": int(person_id),

            "confidence": float(confidence)

        })


    return detections


# ============================================================
# DRAW DETECTIONS
# ============================================================

def draw_detections(
    frame,
    detections
):

    output = frame.copy()


    for detection in detections:

        x1, y1, x2, y2 = detection["box"]

        person_id = detection["id"]

        confidence = detection["confidence"]


        cv2.rectangle(

            output,

            (x1, y1),

            (x2, y2),

            GREEN,

            2

        )


        label = (

            f"ID:{person_id} "
            f"{confidence:.2f}"

        )


        cv2.putText(

            output,

            label,

            (x1, max(y1 - 8, 20)),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            GREEN,

            2

        )


    return output


# ============================================================
# CALCULATE FRAME METRICS
# ============================================================

def calculate_frame_metrics(
    actual,
    detected
):

    difference = detected - actual


    # --------------------------------------------------------
    # Extra detections
    # --------------------------------------------------------

    false_positives = max(
        0,
        difference
    )


    # --------------------------------------------------------
    # Missed people
    # --------------------------------------------------------

    missed = max(
        0,
        -difference
    )


    # --------------------------------------------------------
    # Count accuracy
    # --------------------------------------------------------

    if actual == 0:

        if detected == 0:

            accuracy = 100.0

        else:

            accuracy = 0.0

    else:

        error = abs(
            detected - actual
        )

        accuracy = max(

            0.0,

            100.0
            *
            (
                1
                -
                error / actual
            )

        )


    return (

        false_positives,

        missed,

        accuracy

    )


# ============================================================
# RUN ONE TEST
# ============================================================

def run_test(
    model,
    camera,
    test_number,
    actual_people
):

    print()
    print("=" * 60)
    print(f"TEST {test_number}")
    print("=" * 60)

    print(
        f"Actual people: {actual_people}"
    )

    print(
        f"Frames: {FRAMES_PER_TEST}"
    )

    print()
    print(
        "Keep the same number of people "
        "visible during this test."
    )

    print()
    print(
        "Starting in 3 seconds..."
    )

    time.sleep(3)


    detected_counts = []

    accuracies = []

    false_positive_counts = []

    missed_counts = []

    unique_ids = set()

    duplicate_id_events = 0

    previous_ids = set()


    frame_number = 0


    while frame_number < FRAMES_PER_TEST:

        ret, frame = camera.read()


        if not ret:

            print(
                "Camera frame could not be read."
            )

            break


        detections = get_detections(
            model,
            frame
        )


        detected_count = len(
            detections
        )


        # ----------------------------------------------------
        # IDs currently visible
        # ----------------------------------------------------

        current_ids = set()


        for detection in detections:

            person_id = detection["id"]


            if person_id >= 0:

                current_ids.add(
                    person_id
                )

                unique_ids.add(
                    person_id
                )


        # ----------------------------------------------------
        # Check sudden ID changes
        # ----------------------------------------------------

        if previous_ids and current_ids:

            common_ids = (
                previous_ids
                &
                current_ids
            )


            if (
                len(previous_ids) > 0
                and
                len(current_ids) > 0
                and
                len(common_ids) == 0
            ):

                duplicate_id_events += 1


        previous_ids = current_ids


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        (

            false_positives,

            missed,

            accuracy

        ) = calculate_frame_metrics(

            actual_people,

            detected_count

        )


        detected_counts.append(
            detected_count
        )

        accuracies.append(
            accuracy
        )

        false_positive_counts.append(
            false_positives
        )

        missed_counts.append(
            missed
        )


        # ----------------------------------------------------
        # Draw
        # ----------------------------------------------------

        output = draw_detections(
            frame,
            detections
        )


        # ----------------------------------------------------
        # Information panel
        # ----------------------------------------------------

        cv2.putText(

            output,

            f"TEST: {test_number}",

            (20, 35),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            WHITE,

            2

        )


        cv2.putText(

            output,

            f"ACTUAL PEOPLE: {actual_people}",

            (20, 70),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            BLUE,

            2

        )


        cv2.putText(

            output,

            f"DETECTED: {detected_count}",

            (20, 105),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            GREEN,

            2

        )


        cv2.putText(

            output,

            f"FRAME ACCURACY: {accuracy:.1f}%",

            (20, 140),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            YELLOW,

            2

        )


        cv2.putText(

            output,

            f"FRAME: {frame_number + 1}/{FRAMES_PER_TEST}",

            (20, 175),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            WHITE,

            2

        )


        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        if detected_count == actual_people:

            status = "CORRECT COUNT"

            status_color = GREEN

        elif detected_count > actual_people:

            status = "EXTRA DETECTION"

            status_color = RED

        else:

            status = "MISSED PERSON"

            status_color = RED


        cv2.putText(

            output,

            status,

            (20, 215),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            status_color,

            2

        )


        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        cv2.imshow(

            "AI Crowd Accuracy Test",

            output

        )


        key = cv2.waitKey(1) & 0xFF


        # ESC
        if key == 27:

            print()
            print(
                "Test stopped by user."
            )

            break


        frame_number += 1


        time.sleep(
            SAMPLE_INTERVAL
        )


    # ========================================================
    # TEST RESULTS
    # ========================================================

    if len(detected_counts) == 0:

        return None


    average_detected = statistics.mean(
        detected_counts
    )


    average_accuracy = statistics.mean(
        accuracies
    )


    mae = statistics.mean(

        abs(
            detected - actual_people
        )

        for detected
        in detected_counts

    )


    exact_frames = sum(

        1

        for detected
        in detected_counts

        if detected == actual_people

    )


    exact_percentage = (

        exact_frames
        /
        len(detected_counts)
        *
        100

    )


    total_false_positives = sum(
        false_positive_counts
    )


    total_missed = sum(
        missed_counts
    )


    min_detected = min(
        detected_counts
    )


    max_detected = max(
        detected_counts
    )


    result = {

        "test": test_number,

        "actual": actual_people,

        "average_detected":
            average_detected,

        "min_detected":
            min_detected,

        "max_detected":
            max_detected,

        "average_accuracy":
            average_accuracy,

        "exact_count_percentage":
            exact_percentage,

        "mae":
            mae,

        "false_positives":
            total_false_positives,

        "missed":
            total_missed,

        "unique_ids":
            len(unique_ids),

        "id_change_events":
            duplicate_id_events

    }


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("-" * 60)
    print(f"TEST {test_number} RESULTS")
    print("-" * 60)

    print(
        f"Actual people:           {actual_people}"
    )

    print(
        f"Average detected:        {average_detected:.2f}"
    )

    print(
        f"Minimum detected:        {min_detected}"
    )

    print(
        f"Maximum detected:        {max_detected}"
    )

    print(
        f"Average count accuracy:  "
        f"{average_accuracy:.2f}%"
    )

    print(
        f"Exact-count frames:      "
        f"{exact_percentage:.2f}%"
    )

    print(
        f"Mean absolute error:     "
        f"{mae:.2f}"
    )

    print(
        f"Extra detections:        "
        f"{total_false_positives}"
    )

    print(
        f"Missed people:           "
        f"{total_missed}"
    )

    print(
        f"Unique tracking IDs:     "
        f"{len(unique_ids)}"
    )

    print(
        f"ID-change events:        "
        f"{duplicate_id_events}"
    )

    print("-" * 60)


    return result


# ============================================================
# SAVE CSV
# ============================================================

def save_results(results):

    if not results:

        return


    fieldnames = [

        "test",

        "actual",

        "average_detected",

        "min_detected",

        "max_detected",

        "average_accuracy",

        "exact_count_percentage",

        "mae",

        "false_positives",

        "missed",

        "unique_ids",

        "id_change_events"

    ]


    with open(

        CSV_FILE,

        "w",

        newline=""

    ) as file:

        writer = csv.DictWriter(

            file,

            fieldnames=fieldnames

        )


        writer.writeheader()


        for result in results:

            writer.writerow(
                result
            )


    print()
    print(
        f"Results saved to: {CSV_FILE}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_final_summary(results):

    if not results:

        return


    average_accuracy = statistics.mean(

        result["average_accuracy"]

        for result in results

    )


    average_mae = statistics.mean(

        result["mae"]

        for result in results

    )


    total_false_positives = sum(

        result["false_positives"]

        for result in results

    )


    total_missed = sum(

        result["missed"]

        for result in results

    )


    print()
    print()
    print("=" * 70)
    print("FINAL AI CROWD MONITORING ACCURACY REPORT")
    print("=" * 70)


    print()

    print(
        f"Number of tests:             "
        f"{len(results)}"
    )


    print(
        f"Average count accuracy:      "
        f"{average_accuracy:.2f}%"
    )


    print(
        f"Average MAE:                 "
        f"{average_mae:.2f}"
    )


    print(
        f"Total extra detections:      "
        f"{total_false_positives}"
    )


    print(
        f"Total missed people:         "
        f"{total_missed}"
    )


    print()

    print(
        "Test-by-test:"
    )


    for result in results:

        print(

            f"  Test {result['test']}: "

            f"Actual={result['actual']} | "

            f"Average="
            f"{result['average_detected']:.2f} | "

            f"Accuracy="
            f"{result['average_accuracy']:.2f}% | "

            f"MAE="
            f"{result['mae']:.2f}"

        )


    print()
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("AI CROWD MONITORING - ACCURACY TEST")
    print("=" * 70)

    print()
    print(
        "This program tests person-counting accuracy"
    )

    print(
        "using your current camera and tracker."
    )

    print()

    print(
        f"Frames per test: {FRAMES_PER_TEST}"
    )

    print(
        f"Sample interval: {SAMPLE_INTERVAL} seconds"
    )

    print()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()


    # --------------------------------------------------------
    # Open camera
    # --------------------------------------------------------

    print()
    print("Opening camera...")

    camera = cv2.VideoCapture(
        CAMERA_INDEX
    )


    if not camera.isOpened():

        print()
        print("ERROR: Camera could not be opened.")
        print()
        return


    print(
        "Camera opened successfully."
    )


    # --------------------------------------------------------
    # Test instructions
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TEST INSTRUCTIONS")
    print("=" * 70)

    print()
    print(
        "For each test, enter the REAL number of people"
    )

    print(
        "currently visible in the camera."
    )

    print()
    print(
        "Example:"
    )

    print(
        "If 8 people are actually visible, enter:"
    )

    print(
        "8"
    )

    print()
    print(
        "Keep the number of people stable during"
    )

    print(
        "the 100-frame test."
    )

    print()
    print(
        "Press ESC during a test to stop it."
    )

    print()


    results = []

    test_number = 1


    # ========================================================
    # TEST LOOP
    # ========================================================

    while True:

        try:

            user_input = input(

                f"Enter ACTUAL people count "
                f"for Test {test_number} "
                f"(or Q to finish): "

            ).strip()


        except KeyboardInterrupt:

            print()
            break


        if user_input.lower() == "q":

            break


        try:

            actual_people = int(
                user_input
            )

        except ValueError:

            print(
                "Please enter a whole number."
            )

            continue


        if actual_people < 0:

            print(
                "People count cannot be negative."
            )

            continue


        result = run_test(

            model,

            camera,

            test_number,

            actual_people

        )


        if result is not None:

            results.append(
                result
            )


        test_number += 1


        print()

        again = input(

            "Run another test? "
            "(Y/N): "

        ).strip().lower()


        if again != "y":

            break


    # ========================================================
    # CLEANUP
    # ========================================================

    camera.release()

    cv2.destroyAllWindows()


    # ========================================================
    # SAVE AND DISPLAY
    # ========================================================

    save_results(
        results
    )

    print_final_summary(
        results
    )


    print()
    print(
        "Accuracy testing completed."
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()