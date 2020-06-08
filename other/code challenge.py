def find_meeting(schedule1, boundary1, schedule2, boundary2):
    p1_available = find_availability(schedule1, boundary1)
    p2_available = find_availability(schedule2, boundary2)
    match_list = []

    for time in p1_available:
        p1_start, p1_end = time
        for match in p2_available:
            p2_start, p2_end = match
            if p1_start < p2_end and p1_end > p2_start:
                match_list.append([max(p1_start, p2_start),min(p1_end, p2_end)])
                
    return match_list
            

def find_availability(schedule: list, boundary:list):
    availability = []
    start_bound, end_bound = boundary
    time_index = start_bound
    list_index = 0

    while start_bound <= time_index < end_bound:
        start_meet, end_meet = schedule[list_index]
        if start_meet <= time_index < end_meet:
            time_index += end_meet - start_meet
            list_index += 1
            if list_index = len(schedule):
                break
            continue
        else:
            availability.append([time_index, start_meet])
            time_index += start_meet - time_index

    availability.append([time_index, end_bound])
    return availability
            
            
