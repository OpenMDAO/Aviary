import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.mission.gasp_based.ode.breguet_cruise_eom import E_RangeComp, RangeComp
from aviary.variable_info.variables import Dynamic


class TestBreguetResults(unittest.TestCase):
    """Test cruise range and time in RangeComp component."""

    def setUp(self):
        nn = 10

        self.prob = om.Problem()
        self.prob.model.add_subsystem('range_comp', RangeComp(num_nodes=nn), promotes=['*'])

        self.prob.setup(check=False, force_alloc_complex=True)

        self.prob.set_val('TAS_cruise', 458.8, units='kn')
        self.prob.set_val('mass', np.linspace(171481, 171481 - 10000, nn), units='lbm')
        self.prob.set_val(
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            -5870
            * np.ones(
                nn,
            ),
            units='lbm/h',
        )

    def test_case1(self):
        tol = 1e-6
        self.prob.run_model()

        cruise_range = self.prob.get_val('cruise_range', units='NM')
        cruise_time = self.prob.get_val('cruise_time', units='s')

        t_expected = 6134.7240144
        r_expected = 781.83848643

        assert_near_equal(cruise_range[-1, ...], r_expected, tolerance=0.001)
        assert_near_equal(cruise_time[-1, ...], t_expected, tolerance=0.001)

        with np.printoptions(linewidth=1024):
            self.prob.model.list_outputs(prom_name=True, print_arrays=True)
            partial_data = self.prob.check_partials(method='cs')  # , out_stream=None)
        assert_check_partials(partial_data, atol=tol, rtol=tol)


class TestBreguetPartials(unittest.TestCase):
    def setUp(self):
        nn = 10

        self.prob = om.Problem()
        self.prob.model.add_subsystem('range_comp', RangeComp(num_nodes=nn), promotes=['*'])

        self.prob.model.set_input_defaults(
            'TAS_cruise',
            458.8
            + 50
            * np.random.rand(
                nn,
            ),
            units='kn',
        )
        self.prob.model.set_input_defaults(
            'mass', np.linspace(171481, 171481 - 10000, nn), units='lbm'
        )
        self.prob.model.set_input_defaults(
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            -5870
            * np.ones(
                nn,
            ),
            units='lbm/h',
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_partials(self):
        tol = 1e-10
        self.prob.run_model()

        # cruise_range = self.prob.get_val("cruise_range", units="NM")
        # cruise_time = self.prob.get_val("cruise_time", units="s")

        # t_expected = 6134.7240144
        # r_expected = 781.83848643
        #
        # assert_near_equal(cruise_range[-1, ...], r_expected, tolerance=0.001)
        # assert_near_equal(cruise_time[-1, ...], t_expected, tolerance=0.001)

        with np.printoptions(linewidth=1024):
            self.prob.model.list_outputs(prom_name=True, print_arrays=True)
            partial_data = self.prob.check_partials(method='cs')  # , out_stream=None)
        assert_check_partials(partial_data, atol=tol, rtol=tol)


class TestBreguetPartials2(unittest.TestCase):
    """Test mass-weight conversion."""

    def setUp(self):
        import aviary.mission.gasp_based.ode.breguet_cruise_eom as breguet

        breguet.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.mission.gasp_based.ode.breguet_cruise_eom as breguet

        breguet.GRAV_ENGLISH_LBM = 1.0

    def test_partials(self):
        nn = 2
        prob = om.Problem()
        prob.model.add_subsystem('range_comp', RangeComp(num_nodes=nn), promotes=['*'])
        prob.model.set_input_defaults(
            'TAS_cruise',
            458.8
            + 50
            * np.random.rand(
                nn,
            ),
            units='kn',
        )
        prob.model.set_input_defaults('mass', np.linspace(171481, 171481 - 10000, nn), units='lbm')
        prob.model.set_input_defaults(
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            -5870
            * np.ones(
                nn,
            ),
            units='lbm/h',
        )
        prob.setup(check=False, force_alloc_complex=True)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=2e-11, rtol=1e-11)


class TestBreguetResults2(unittest.TestCase):
    def setUp(self):
        self.nn = nn = 100

        self.prob = om.Problem()
        self.prob.model.add_subsystem('range_comp', RangeComp(num_nodes=nn), promotes=['*'])

        self.prob.setup(check=False, force_alloc_complex=True)

        self.prob.set_val('TAS_cruise', 458.8, units='kn')
        self.prob.set_val('mass', np.linspace(171481, 171481 - 10000, nn), units='lbm')
        self.prob.set_val(
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            -5870
            * np.ones(
                nn,
            ),
            units='lbm/h',
        )

    def test_results(self):
        self.prob.run_model()

        W = self.prob.get_val('mass', units='lbm') * GRAV_ENGLISH_LBM
        V = self.prob.get_val('TAS_cruise', units='kn')
        r = self.prob.get_val('cruise_range', units='NM')
        t = self.prob.get_val('cruise_time', units='h')
        fuel_flow = -self.prob.get_val(
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL, units='lbm/h'
        )

        v_avg = (V[:-1] + V[1:]) / 2
        fuel_flow_avg = (fuel_flow[:-1] + fuel_flow[1:]) / 2

        # Range should be equal to the product of initial speed in the segment and change in time
        assert_near_equal(np.diff(r), v_avg * np.diff(t), tolerance=1.0e-5)

        # time should satisfy: dt = -dW / fuel_flow
        assert_near_equal(np.diff(t), -np.diff(W) / fuel_flow_avg, tolerance=1.0e-6)


class TestElectricBreguetResults(unittest.TestCase):
    """Test cruise range and time in E_RangeComp component."""

    def setUp(self):
        nn = 10

        self.prob = om.Problem()
        self.prob.model.add_subsystem('e_range_comp', E_RangeComp(num_nodes=nn), promotes=['*'])

        self.prob.setup(check=False, force_alloc_complex=True)

        self.prob.set_val('TAS_cruise', 458.8, units='kn')
        self.prob.set_val(
            Dynamic.Vehicle.CUMULATIVE_ELECTRIC_ENERGY_USED,
            np.linspace(5843, 19390, nn),
            units='kW*h',
        )
        self.prob.set_val(
            Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN_TOTAL,
            7531.64
            * np.ones(
                nn,
            ),
            units='kW',
        )  # 10100.1 hp in GASP

    def test_case1(self):
        tol = 1e-6
        self.prob.run_model()

        cruise_range = self.prob.get_val('cruise_range', units='NM')
        cruise_time = self.prob.get_val('cruise_time', units='s')

        t_expected = 6475.243
        r_expected = 825.2337

        assert_near_equal(cruise_range[-1, ...], r_expected, tolerance=tol)
        assert_near_equal(cruise_time[-1, ...], t_expected, tolerance=tol)

    def test_partials(self):
        tol = 1e-10
        self.prob.run_model()

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=tol, rtol=tol)


if __name__ == '__main__':
    # unittest.main()
    test = TestBreguetPartials2()
    test.setUp()
    test.test_partials()
