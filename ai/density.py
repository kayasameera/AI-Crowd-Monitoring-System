def calculate_density(current_people):

    if current_people <= 3:
        return "LOW"

    elif current_people <= 7:
        return "MEDIUM"

    else:
        return "HIGH"