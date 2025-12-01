import numpy as np
from openmdao.utils.assert_utils import assert_near_equal

from aviary.utils.test_utils.assert_utils import warn_timeseries_near_equal


def compare_against_expected_values(prob, expected_dict):
    """Compare values in prob with the ones in expected_dict."""
    expected_times = expected_dict['times']
    expected_altitudes = expected_dict['altitudes']
    expected_masses = expected_dict['masses']
    expected_ranges = expected_dict['ranges']
    expected_velocities = expected_dict['velocities']

    times = []
    altitudes = []
    masses = []
    ranges = []
    velocities = []

    for idx, phase in enumerate(['climb', 'cruise', 'descent']):
        times.extend(prob.get_val(f'traj.{phase}.timeseries.time', units='s', get_remote=True))
        altitudes.extend(
            prob.get_val(f'traj.{phase}.timeseries.altitude', units='m', get_remote=True)
        )
        velocities.extend(
            prob.get_val(f'traj.{phase}.timeseries.velocity', units='m/s', get_remote=True)
        )
        masses.extend(prob.get_val(f'traj.{phase}.timeseries.mass', units='kg', get_remote=True))
        ranges.extend(prob.get_val(f'traj.{phase}.timeseries.distance', units='m', get_remote=True))

    times = np.array(times)
    altitudes = np.array(altitudes)
    masses = np.array(masses)
    ranges = np.array(ranges)
    velocities = np.array(velocities)

    # Check Objective and other key variables to a reasonably tight tolerance.

    rtol = 2.0e-2

    # Mass at the end of Descent
    assert_near_equal(masses[-1], expected_masses[-1], tolerance=rtol)

    # Range at the end of Descent
    assert_near_equal(ranges[-1], expected_ranges[-1], tolerance=rtol)

    # Flight time
    assert_near_equal(times[-1], expected_times[-1], tolerance=rtol)

    # Altitude at the start and end of climb
    # Added because setting up phase options is a little tricky.
    assert_near_equal(altitudes[0], expected_altitudes[0], tolerance=rtol)
    assert_near_equal(altitudes[-1], expected_altitudes[-1], tolerance=rtol)

    # Check mission values.

    # NOTE rtol = 0.05 = 5% different from truth (first timeseries)
    #      atol = 2 = no more than +/-2 meter/second/kg difference between values
    #      atol_altitude - 30 ft.
    rtol = 0.05
    atol = 2.0
    atol_altitude = 30.0

    # FLIGHT PATH
    warn_timeseries_near_equal(
        times,
        altitudes,
        expected_times,
        expected_altitudes,
        abs_tolerance=atol_altitude,
        rel_tolerance=rtol,
    )
    warn_timeseries_near_equal(
        times, masses, expected_times, expected_masses, abs_tolerance=atol, rel_tolerance=rtol
    )
    warn_timeseries_near_equal(
        times, ranges, expected_times, expected_ranges, abs_tolerance=atol, rel_tolerance=rtol
    )
    warn_timeseries_near_equal(
        times,
        velocities,
        expected_times,
        expected_velocities,
        abs_tolerance=atol,
        rel_tolerance=rtol,
    )
