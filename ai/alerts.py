OVER_CROWD_LIMIT = 10


def check_overcrowding(current_people):

    if current_people >= OVER_CROWD_LIMIT:
        return True

    return False