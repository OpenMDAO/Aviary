import math

import numpy as np


def distributed_engine_count_factor(total_num_eng: int) -> float:
    """
    Returns the distributed propulsion engine count factor.

    Parameters
    ----------
    total_num_eng : int
        Total number of engines
    """
    factor = float(total_num_eng)

    if total_num_eng > 4:
        factor = 4.0 + 2.0 * math.atan((total_num_eng - 4.0) / 3.0)

    return factor


def distributed_thrust_factor(max_sls_thrust: float, total_num_eng: int) -> float:
    """
    Returns the distributed propulsion thrust factor.

    Parameters
    ----------
    max_sls_thrust : float
        Total maximum SLS thrust
    total_num_eng : iterable or int
        Total number of engines
    """
    num_engine_factor = distributed_engine_count_factor(total_num_eng)
    return max_sls_thrust / num_engine_factor


def distributed_nacelle_diam_factor(diam_nacelle: list, num_eng: list) -> float:
    """
    Returns the distributed propulsion nacelle average diameter factor.
    If more than one engine type are present on the aircraft, use the average of all
    nacelle diameters (accounting for how many of each engine type are present).

    Parameters
    ----------
    diam_nacelle : iterable or float
        Nacelle average diameter for each engine model
    num_eng : iterable or int
        Number of engines for each engine model
    """
    # If there is more than one engine model, use the global average diameter
    try:
        total_num_eng = sum(num_eng)
        diam_avg = sum(diam_nacelle * num_eng) / total_num_eng
    except TypeError:
        total_num_eng = num_eng
        diam_avg = diam_nacelle

    diam_factor = diam_avg
    if total_num_eng > 4:
        diam_factor = 0.5 * diam_avg * total_num_eng**0.5

    return diam_factor


def distributed_nacelle_diam_factor_deriv(num_eng: int) -> float:
    """
    Returns the derivative of the distributed propulsion nacelle average diameter factor
    w.r.t. the global nacelle average diameter.

    Parameters
    ----------
    num_eng : iterable or int
        Number of engines for each engine model
    """
    try:
        total_num_engines = sum(num_eng)
    except TypeError:
        total_num_engines = num_eng

    deriv = 1.0
    if total_num_engines > 4:
        deriv = num_eng * (0.5 * total_num_engines**0.5) / total_num_engines

    return deriv


def nacelle_count_factor(num_eng):
    """
    Returns the nacelle count factor, which is the number of engines plus
    0.5 if there is a centerline engine. It is assumed there is a centerline
    engine if the number of engines is odd (for each unique engine type).

    If there are multiple engine types, each engine type gets its own nacelle count
    factor. This methodology does not account for potential conflict of engines present
    on the aircraft centerline.

    Parameters
    ----------
    num_eng : iterable or int
        Number of engines for each engine model
    """
    try:
        return np.array([n + n % 2 * 0.5 for n in num_eng])
    except TypeError:
        return np.array([num_eng + num_eng % 2 * 0.5])
