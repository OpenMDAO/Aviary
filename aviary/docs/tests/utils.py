import inspect


def check_value(val1, val2):
    if isinstance(val1, (str, int, float, list, tuple, dict)):
        if val1 != val2:
            raise ValueError(f"{val1} is not equal to {val2}")
    else:
        if val1 is not val2:
            raise ValueError(f"{val1} is not {val2}")


def check_args(func, expected_args: list, args_to_ignore=['self']):
    args = [arg for arg in inspect.getfullargspec(func)[0] if arg not in args_to_ignore]
    check_value(args.sort(), expected_args.sort())
