def update_occupancy(previous_entry_count,
                     previous_exit_count,
                     entry_count,
                     exit_count,
                     occupancy):

    # New entry detected
    if entry_count > previous_entry_count:
        occupancy += entry_count - previous_entry_count

    # New exit detected
    if exit_count > previous_exit_count:
        occupancy -= exit_count - previous_exit_count

    # Occupancy should never be negative
    occupancy = max(0, occupancy)

    return occupancy