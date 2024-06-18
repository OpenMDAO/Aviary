def check_value(val1, val2):
    if val1 != val2:
        raise ValueError(f"{val1} is not {val2}")
