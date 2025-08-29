import math

def linear_schedule_with_min(initial_value, min_value):
    def func(progress_remaining):
        return min_value + (initial_value - min_value) * progress_remaining
    return func