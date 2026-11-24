import math


# ============================================================
# SETTINGS
# ============================================================

# Minimum YOLO confidence required
MIN_CONFIDENCE = 0.55

# Number of consecutive frames required before
# a new person is confirmed as an entry
CONFIRM_FRAMES = 5

# Number of frames a person can disappear before
# being considered exited
MAX_MISSING_FRAMES = 60

# Maximum distance in pixels to match a new YOLO ID
# with an already known person
MATCH_DISTANCE = 120


# ============================================================
# DISTANCE FUNCTION
# ============================================================

def distance(point1, point2):

    return math.sqrt(

        (point1[0] - point2[0]) ** 2
        +
        (point1[1] - point2[1]) ** 2

    )


# ============================================================
# ENTRY / EXIT COUNTING
# ============================================================

def count_entry_exit(
    results,
    previous_positions,
    entry_count,
    exit_count
):

    # --------------------------------------------------------
    # Create tracking storage
    # --------------------------------------------------------

    if "people" not in previous_positions:

        previous_positions["people"] = {}

    people = previous_positions["people"]


    # --------------------------------------------------------
    # Get current detections
    # --------------------------------------------------------

    current_detections = []


    if results[0].boxes.id is not None:

        boxes = (
            results[0]
            .boxes
            .xyxy
            .cpu()
            .numpy()
        )

        ids = (
            results[0]
            .boxes.id
            .cpu()
            .numpy()
        )

        # Confidence scores
        if results[0].boxes.conf is not None:

            confidences = (
                results[0]
                .boxes.conf
                .cpu()
                .numpy()
            )

        else:

            confidences = [1.0] * len(ids)


        # Class IDs
        if results[0].boxes.cls is not None:

            classes = (
                results[0]
                .boxes.cls
                .cpu()
                .numpy()
            )

        else:

            classes = [0] * len(ids)


        for box, person_id, confidence, class_id in zip(

            boxes,
            ids,
            confidences,
            classes

        ):

            # ------------------------------------------------
            # Only PERSON class
            # COCO person class = 0
            # ------------------------------------------------

            if int(class_id) != 0:

                continue


            # ------------------------------------------------
            # Confidence filter
            # ------------------------------------------------

            if float(confidence) < MIN_CONFIDENCE:

                continue


            x1, y1, x2, y2 = map(

                int,

                box

            )


            center_x = (x1 + x2) // 2

            center_y = (y1 + y2) // 2


            current_detections.append({

                "id": int(person_id),

                "center": (
                    center_x,
                    center_y
                ),

                "confidence": float(confidence)

            })


    # --------------------------------------------------------
    # Track which internal people were seen this frame
    # --------------------------------------------------------

    matched_people = set()


    # ========================================================
    # MATCH CURRENT DETECTIONS
    # ========================================================

    for detection in current_detections:

        yolo_id = detection["id"]

        center = detection["center"]


        best_person = None

        best_distance = MATCH_DISTANCE


        # ----------------------------------------------------
        # Try to match with an already known person
        # ----------------------------------------------------

        for internal_id, person in people.items():

            if internal_id in matched_people:

                continue


            if not person["active"]:

                continue


            d = distance(

                center,

                person["center"]

            )


            if d < best_distance:

                best_distance = d

                best_person = internal_id


        # ====================================================
        # EXISTING PERSON
        # ====================================================

        if best_person is not None:

            person = people[best_person]


            # Update position

            person["center"] = center

            person["yolo_id"] = yolo_id

            person["missing"] = 0

            person["frames_seen"] += 1

            matched_people.add(best_person)


            # ------------------------------------------------
            # Confirm the person
            # ------------------------------------------------

            if (

                not person["confirmed"]

                and

                person["frames_seen"]
                >= CONFIRM_FRAMES

            ):

                person["confirmed"] = True

                entry_count += 1


                print(

                    f"ENTRY -> Person {best_person} | "

                    f"Total Entry: {entry_count}"

                )


        # ====================================================
        # NEW POSSIBLE PERSON
        # ====================================================

        else:

            # Create a stable internal ID

            if len(people) == 0:

                internal_id = 1

            else:

                internal_id = max(

                    people.keys()

                ) + 1


            people[internal_id] = {

                "center": center,

                "yolo_id": yolo_id,

                "missing": 0,

                "frames_seen": 1,

                "confirmed": False,

                "active": True

            }


            matched_people.add(internal_id)


    # ========================================================
    # UPDATE MISSING PEOPLE
    # ========================================================

    for internal_id in list(people.keys()):

        person = people[internal_id]


        if internal_id not in matched_people:

            person["missing"] += 1


            # ------------------------------------------------
            # Person disappeared
            # ------------------------------------------------

            if (

                person["missing"]
                >= MAX_MISSING_FRAMES

                and

                person["active"]

            ):

                person["active"] = False


                # Only count exit if the person was
                # previously confirmed as an entry

                if person["confirmed"]:

                    exit_count += 1


                    print(

                        f"EXIT -> Person {internal_id} | "

                        f"Total Exit: {exit_count}"

                    )


    # ========================================================
    # RETURN
    # ========================================================

    return (

        previous_positions,

        entry_count,

        exit_count

    )