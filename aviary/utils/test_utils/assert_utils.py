"""
Utilities used for testing Aviary code.
"""
from dymos.utils.testing_utils import assert_timeseries_near_equal


def warn_timeseries_near_equal(t_ref, x_ref, t_check, x_check, abs_tolerance=None,
                               rel_tolerance=None):
    """
    Wraps timeseries check with a try block that prints a warning if the assert fails.
    """
    try:
        assert_timeseries_near_equal(t_ref, x_ref, t_check, x_check, abs_tolerance=abs_tolerance,
                                     rel_tolerance=rel_tolerance)
    except AssertionError as exc:
        print("Warning:\n", str(exc))
