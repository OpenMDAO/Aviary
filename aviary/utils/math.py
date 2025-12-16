"""
Smooth functions and their derivatives.
"""
import numpy as np


def sigmoidX(x, x0, mu=1.0):
    """
    Sigmoid used to smoothly transition between piecewise functions.

    Parameters
    ----------
    x: float or array
        independent variable
    x0: float
        the center of symmetry. When x = x0, sigmoidX = 1/2.
    mu: float
        steepness parameter.

    Returns
    -------
    float or array
        smoothed value from input parameter x.
    """
    if mu == 0:
        raise ValueError('mu must be non-zero')

    if isinstance(x, np.ndarray):
        if np.isrealobj(x):
            dtype = float
        else:
            dtype = complex
        n_size = x.size
        y = np.zeros(n_size, dtype=dtype)
        # avoid overflow in squared term, underflow seems to be ok
        calc_idx = np.where((x.real - x0) / mu > -320)
        y[calc_idx] = 1 / (1 + np.exp(-(x[calc_idx] - x0) / mu))
    else:
        if isinstance(x, float):
            dtype = float
        else:
            dtype = complex
        y = 0
        if (x - x0) * mu > -320:
            y = 1 / (1 + np.exp(-(x - x0) / mu))
    if dtype == float:
        y = y.real
    return y


def dSigmoidXdx(x, x0, mu=1.0):
    """
    Derivative of sigmoid function.

    Parameters
    ----------
    x: float or array
        independent variable
    x0: float
        the center of symmetry. When x = x0, sigmoidX = 1/2.
    mu: float
        steepness parameter.

    Returns
    -------
    float or array
        smoothed derivative value from input parameter x.
    """
    if mu == 0:
        raise ValueError('mu must be non-zero')

    if isinstance(x, np.ndarray):
        if np.isrealobj(x):
            dtype = float
        else:
            dtype = complex
        n_size = x.size
        y = np.zeros(n_size, dtype=dtype)
        term = np.zeros(n_size, dtype=dtype)
        term2 = np.zeros(n_size, dtype=dtype)
        # avoid overflow in squared term, underflow seems to be ok
        calc_idx = np.where((x.real - x0) / mu > -320)
        term[calc_idx] = np.exp(-(x[calc_idx] - x0) / mu)
        term2[calc_idx] = (1 + term[calc_idx]) * (1 + term[calc_idx])
        y[calc_idx] = term[calc_idx] / mu / term2[calc_idx]
    else:
        y = 0
        if (x - x0) * mu > -320:
            term = np.exp(-(x - x0) / mu)
            term2 = (1 + term) * (1 + term)
            y = term / mu / term2
    if dtype == float:
        y = y.real
    return y


def smooth_min(x, b, mu=100.0):
    """
    Smooth approximation of the min function using the log-sum-exp trick.

    Parameters:
    x (float or array-like): First value.
    b (float or array-like): Second value.
    mu (float): The smoothing factor. Higher values make it closer to the true minimum. Try between 75 and 275.

    Returns:
    float or array-like: The smooth approximation of min(x, b).
    """
    sum_log_exp = np.log(np.exp(np.multiply(-mu, x)) + np.exp(np.multiply(-mu, b)))
    rv = -(1 / mu) * sum_log_exp
    return rv


def d_smooth_min(x, b, mu=100.0):
    """
    Derivative of function smooth_min(x)

    Parameters:
    x (float or array-like): First value.
    b (float or array-like): Second value.
    mu (float): The smoothing factor. Higher values make it closer to the true minimum. Try between 75 and 275.

    Returns:
    float or array-like: The smooth approximation of derivative of min(x, b).
    """
    d_sum_log_exp = np.exp(np.multiply(-mu, x)) / (
        np.exp(np.multiply(-mu, x)) + np.exp(np.multiply(-mu, b))
    )
    return d_sum_log_exp


def smooth_max(x, b, mu=10.0):
    """
    Smooth approximation of the min function using the log-sum-exp trick.

    Parameters:
    x (float or array-like): First value.
    b (float or array-like): Second value.
    mu (float): The smoothing factor. Higher values make it closer to the true maximum. Try between 75 and 275.

    Returns:
    float or array-like: The smooth approximation of max(x, b).
    """
    mu_x = mu * x
    mu_b = mu * b
    m = np.maximum(mu_x, mu_b)
    sum_log_exp = (m + np.log(np.exp(mu_x - m) + np.exp(mu_b - m))) / mu

    return sum_log_exp


def d_smooth_max(x, b, mu=10.0):
    """
    Derivative of function smooth_min(x)

    Parameters:
    x (float or array-like): First value.
    b (float or array-like): Second value.
    mu (float): The smoothing factor. Higher values make it closer to the true minimum. Try between 75 and 275.

    Returns:
    float or array-like: The smooth approximation of derivative of min(x, b).
    """
    mu_x = mu * x
    mu_b = mu * b
    m = np.maximum(mu_x, mu_b)
    numerator = np.exp(mu_x - m)
    denominator = np.exp(mu_x - m) + np.exp(mu_b - m)
    d_sum_log_exp = mu * numerator / denominator
    return d_sum_log_exp


def sin_int4(val):
    """Define a smooth, differentialbe approximation to the 'int' function."""
    return sin_int(sin_int(sin_int(sin_int(val)))) - 0.5


def dydx_sin_int4(val):
    """Define the derivative (dy/dx) of sin_int4, at x = val."""
    y0 = sin_int(val)
    y1 = sin_int(y0)
    y2 = sin_int(y1)

    dydx3 = dydx_sin_int(y2)
    dydx2 = dydx_sin_int(y1)
    dydx1 = dydx_sin_int(y0)
    dydx0 = dydx_sin_int(val)

    dydx = dydx3 * dydx2 * dydx1 * dydx0

    return dydx


# 'int' function can be approximated by recursively applying this sin function
# which makes a smooth, differentialbe function (is there a good one?)
def sin_int(val):
    """
    Define one step in approximating the 'int' function with a smooth,
    differentialbe function.
    """
    int_val = val - np.sin(2 * np.pi * (val + 0.5)) / (2 * np.pi)

    return int_val


def dydx_sin_int(val):
    """Define the derivative (dy/dx) of sin_int, at x = val."""
    dydx = 1.0 - np.cos(2 * np.pi * (val + 0.5))

    return dydx


def smooth_int_tanh(x, mu=10.0):
    """
    Smooth approximation of int(x) using tanh.
    """
    f = np.floor(x)
    frac = x - f
    t = np.tanh(mu * (frac - 0.5))
    s = 0.5 * (t + 1)
    y = f + s
    return y


def d_smooth_int_tanh(x, mu=10.0):
    """
    Smooth approximation of int(x) using tanh.
    Returns (y, dy_dx).
    """
    f = np.floor(x)
    frac = x - f
    t = np.tanh(mu * (frac - 0.5))
    dy_dx = 0.5 * mu * (1 - t**2)
    return dy_dx
