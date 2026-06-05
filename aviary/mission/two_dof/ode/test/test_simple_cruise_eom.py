import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.mission.two_dof.ode.simple_cruise_eom import DistanceComp
from aviary.variable_info.variables import Dynamic


class TestSimpleCruiseResults(unittest.TestCase):
    """Test cruise with DistanceComp component."""

    def setUp(self):
        nn = 10

        self.prob = om.Problem()
        self.prob.model.add_subsystem('range_comp', DistanceComp(num_nodes=nn), promotes=['*'])

        self.prob.setup(check=False, force_alloc_complex=True)

        self.prob.set_val('TAS_cruise', 458.8, units='kn')
        self.prob.set_val('cruise_distance_initial', 0.0, units='NM')
        self.prob.set_val('time', np.arange(nn) * 6134.72 / 9, units='s')

    def test_case1(self):
        tol = 1e-6
        self.prob.run_model()

        distance = self.prob.get_val(Dynamic.Mission.DISTANCE, units='NM')

        r_expected = 781.838598222

        assert_near_equal(distance[-1, ...], r_expected, tolerance=0.001)


class TestDistanceCompPartials(unittest.TestCase):
    def setUp(self):
        nn = 10

        self.prob = om.Problem()
        self.prob.model.add_subsystem('range_comp', DistanceComp(num_nodes=nn), promotes=['*'])

        self.prob.model.set_input_defaults(
            'TAS_cruise',
            458.8
            + 50
            * np.random.rand(
                nn,
            ),
            units='kn',
        )
        self.prob.setup(check=False, force_alloc_complex=True)

        self.prob.set_val('cruise_distance_initial', 1.0, units='NM')
        self.prob.set_val('time', np.arange(nn) * 6134.72 / 9, units='s')

    def test_partials(self):
        tol = 1e-10
        self.prob.run_model()

        with np.printoptions(linewidth=1024):
            self.prob.model.list_outputs(prom_name=True, print_arrays=True)
            partial_data = self.prob.check_partials(method='cs', out_stream=None)
        assert_check_partials(partial_data, atol=tol, rtol=tol)


if __name__ == '__main__':
    # unittest.main()
    test = TestSimpleCruiseResults()
    test.setUp()
    test.test_case1()
