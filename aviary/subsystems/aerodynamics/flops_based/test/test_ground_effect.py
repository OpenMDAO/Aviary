import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.aerodynamics.flops_based.ground_effect import GroundEffect
from aviary.utils.aviary_values import AviaryValues, get_items
from aviary.variable_info.variables import Aircraft, Dynamic


class TestGroundEffect(unittest.TestCase):
    """
    Perform regression test in all three computational ranges:
    - in ground effect (on the ground);
    - in transition (in the air, but near enough to the ground for limited effect);
    - in free flight (no ground effect).
    """

    def test_regression(self):
        prob: om.Problem = make_problem()

        prob.run_model()

        tol = 1e-12

        for key, (desired, units) in get_items(_regression_data):
            try:
                actual = prob.get_val(key, units)

                assert_near_equal(actual, desired, tol)

            except ValueError as error:
                msg = f'"{key}": {error!s}'

                raise ValueError(msg) from None

        partials = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partials, atol=1e-10, rtol=1e-12)

        assert_near_equal(prob.get_val('lift_coefficient'), [0.558234, 1.200676, 2.0], 1e-6)
        assert_near_equal(prob.get_val('drag_coefficient'), [0.02036, 0.0223957, 0.08], 1e-6)


def make_problem():
    """
    Return a problem that covers all three computational ranges:
    - in ground effect (on the ground);
    - in transition (in the air, but near enough to the ground for limited effect);
    - in free flight (no ground effect).
    """
    nn = 3
    ground_altitude = 100.0  # units='m'

    minimum_drag_coefficient = (0.02036, 'unitless')
    base_lift_coefficient = (np.array([0.5, 1.2, 2.0]), 'unitless')
    base_drag_coefficient = (np.array([0.02036, 1.1 * 0.02036, 0.08]), 'unitless')
    aspect_ratio = (9.45, 'unitless')
    height = (8.0, 'ft')
    span = (34.0, 'm')

    inputs = AviaryValues(
        {
            Dynamic.Vehicle.ANGLE_OF_ATTACK: (np.array([0.0, 2.0, 6]), 'deg'),
            Dynamic.Mission.ALTITUDE: (np.array([100.0, 132, 155]), 'm'),
            Dynamic.Mission.FLIGHT_PATH_ANGLE: (np.array([0.0, 0.5, 1.0]), 'deg'),
            'minimum_drag_coefficient': minimum_drag_coefficient,
            'base_lift_coefficient': base_lift_coefficient,
            'base_drag_coefficient': base_drag_coefficient,
            Aircraft.Wing.ASPECT_RATIO: aspect_ratio,
            Aircraft.Wing.HEIGHT: height,
            Aircraft.Wing.SPAN: span,
        }
    )

    ground_effect = GroundEffect(num_nodes=nn, ground_altitude=ground_altitude)

    prob = om.Problem()
    prob.model.add_subsystem('ground_effect', ground_effect, promotes=['*'])
    prob.setup(force_alloc_complex=True)

    for key, (val, units) in inputs:
        prob.set_val(key, val, units)

    return prob


# NOTE:
# - results from `generate_regression_data()`
#    - last generated 2023 June 7
# - generate new regression data if, and only if, ground effect is updated with a more
#   trusted implementation
_regression_data = AviaryValues(
    dict(
        lift_coefficient=([0.5582336522780803, 1.200675778095871, 2.0], 'unitless'),
        drag_coefficient=([0.02036, 0.022395716686151382, 0.08], 'unitless'),
    )
)


def generate_regression_data():
    """
    Generate regression data that covers all three computational ranges:
    - in ground effect (on the ground);
    - in transition (in the air, but near enough to the ground for limited effect);
    - in free flight (no ground effect).

    Notes
    -----
    Use this function to generate new regression data if, and only if, ground effect is
    updated with a more trusted implementation.
    """
    prob: om.Problem = make_problem()

    prob.run_model()

    lift_coefficient = prob.get_val('lift_coefficient')
    drag_coefficient = prob.get_val('drag_coefficient')

    prob.check_partials(compact_print=True, method='cs')

    print('lift_coefficient', *lift_coefficient, sep=', ')
    print('drag_coefficient', *drag_coefficient, sep=', ')


if __name__ == '__main__':
    unittest.main()
