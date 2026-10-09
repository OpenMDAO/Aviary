import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.mass.mass_to_weight import MassToWeight
from aviary.variable_info.variables import Mission

GRAV_METRIC = 9.80665


class MassToWeightTest(unittest.TestCase):
    """Test computation of weight from mass."""

    def test_case(self):
        prob = om.Problem()

        prob.model.add_subsystem(
            'calc_weight',
            MassToWeight(**{Mission.GRAVITY: (GRAV_METRIC, 'm/s**2')}),
            promotes=['mass', 'weight'],
        )

        prob.setup(force_alloc_complex=True)

        prob.set_val('mass', 120_000, units='kg')

        prob.run_model()
        assert_near_equal(prob.get_val('weight', units='N'), 120_000 * GRAV_METRIC, 1.0e-10)
        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-12)

        nn = 3
        mass = np.full(nn, 120_000)

        prob = om.Problem()

        prob.model.add_subsystem(
            'calc_weight',
            MassToWeight(num_nodes=nn, **{Mission.GRAVITY: (GRAV_METRIC, 'm/s**2')}),
            promotes=['mass', 'weight'],
        )

        prob.setup(force_alloc_complex=True)

        prob.set_val('mass', mass, units='kg')

        prob.run_model()
        assert_near_equal(prob.get_val('weight', units='N'), mass * GRAV_METRIC, 1.0e-10)
        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
