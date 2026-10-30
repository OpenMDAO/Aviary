import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.mission.two_dof.ode.simple_cruise_eom import DistanceComp
from aviary.variable_info.variables import Dynamic


@use_tempdirs
class DistanceCompTestCase(unittest.TestCase):
    """Test the DistanceComp component."""

    def test_results(self):
        nn = 10

        prob = om.Problem()
        prob.model.add_subsystem('range_comp', DistanceComp(num_nodes=nn), promotes=['*'])

        prob.setup(check=False, force_alloc_complex=True)

        prob.set_val('TAS_cruise', 458.8, units='kn')
        prob.set_val('cruise_distance_initial', 0.0, units='NM')
        prob.set_val('time', np.arange(nn) * 6134.72 / 9, units='s')

        prob.run_model()

        distance = prob.get_val(Dynamic.Mission.DISTANCE, units='NM')

        r_expected = 781.838598222

        assert_near_equal(distance[-1, ...], r_expected, tolerance=0.001)

        partials = prob.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials, atol=1e-10, rtol=1e-10)


if __name__ == '__main__':
    unittest.main()
