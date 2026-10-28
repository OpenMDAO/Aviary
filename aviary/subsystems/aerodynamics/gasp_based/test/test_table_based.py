import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.aerodynamics.gasp_based.table_based import (
    GearDragIncrement,
    TabularCruiseAero,
    TabularLowSpeedAero,
)
from aviary.utils.functions import get_aviary_resource_path
from aviary.variable_info.variables import Aircraft, Dynamic

LARGE_SINGLE_AISLE_FREE_DATA = get_aviary_resource_path(
    'models/large_single_aisle_1/aerodynamics_tables/large_single_aisle_1_aero_free.csv'
)
BWB_FREE_DATA = get_aviary_resource_path(
    'models/aircraft/blended_wing_body/aerodynamics_tables/generic_BWB_GASP_aero.csv'
)


class TabularCruiseAeroTestCase(unittest.TestCase):
    """Tests for TabularCruiseAero using free-air lift/drag coefficient tables."""

    def _make_prob(self, num_nodes, aero_data, mach, alpha, altitude):
        prob = om.Problem()
        prob.model = TabularCruiseAero(
            num_nodes=num_nodes,
            aero_data=aero_data,
            connect_training_data=False,
            structured=True,
            extrapolate=True,
        )
        prob.setup(force_alloc_complex=True)

        prob.set_val(Dynamic.Atmosphere.MACH, mach, units='unitless')
        prob.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, alpha, units='deg')
        prob.set_val(Dynamic.Mission.ALTITUDE, altitude, units='ft')
        return prob

    def test_climb(self):
        prob = self._make_prob(
            num_nodes=8,
            aero_data=LARGE_SINGLE_AISLE_FREE_DATA,
            mach=[0.381, 0.384, 0.391, 0.399, 0.8, 0.8, 0.8, 0.8],
            alpha=[5.19, 5.19, 5.19, 5.18, 3.58, 3.81, 4.05, 4.18],
            altitude=[500, 1000, 2000, 3000, 35000, 36000, 37000, 37500],
        )
        prob.run_model()

        expected_values = {
            Dynamic.Vehicle.LIFT_COEFFICIENT: (
                np.array([0.5968, 0.5975, 0.5974, 0.5974, 0.5566, 0.5833, 0.6113, 0.6257]),
                'unitless',
            ),
            Dynamic.Vehicle.DRAG_COEFFICIENT: (
                np.array([0.0307, 0.0307, 0.0307, 0.0307, 0.0296, 0.0310, 0.0326, 0.0334]),
                'unitless',
            ),
        }
        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tolerance=0.009)

        partial_data = prob.check_partials(method='cs', out_stream=None)
        assert_check_partials(partial_data, atol=4e-7, rtol=2e-7)

    def test_cruise(self):
        prob = self._make_prob(
            num_nodes=2,
            aero_data=LARGE_SINGLE_AISLE_FREE_DATA,
            mach=[0.8, 0.8],
            alpha=[4.216, 3.146],
            altitude=[37500, 37500],
        )
        prob.run_model()

        cl_exp = np.array([0.6304, 0.5059])
        cd_exp = cl_exp / np.array([18.608, 18.425])
        expected_values = {
            Dynamic.Vehicle.LIFT_COEFFICIENT: (cl_exp, 'unitless'),
            Dynamic.Vehicle.DRAG_COEFFICIENT: (cd_exp, 'unitless'),
        }
        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tolerance=0.005)

        partial_data = prob.check_partials(method='cs', out_stream=None)
        assert_check_partials(partial_data, atol=9e-8, rtol=2e-7)

    def test_bwb_climb(self):
        # Table modified for demonstration purposes only; does not represent an actual result.
        prob = self._make_prob(
            num_nodes=3,
            aero_data=BWB_FREE_DATA,
            mach=[0.7, 0.8, 0.82],
            alpha=[5.0, 10.0, 2.0],
            altitude=[10000.0, 30000.0, 35000],
        )
        prob.run_model()

        expected_values = {
            Dynamic.Vehicle.LIFT_COEFFICIENT: (
                np.array([0.1509, 0.410764, -0.0384316]),
                'unitless',
            ),
            Dynamic.Vehicle.DRAG_COEFFICIENT: (
                np.array([0.00610224, 0.0205816, 0.00460256]),
                'unitless',
            ),
        }
        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tolerance=0.001)

        partial_data = prob.check_partials(method='cs', out_stream=None)
        assert_check_partials(partial_data, atol=4e-7, rtol=2e-7)

    def test_bwb_cruise(self):
        # Table modified for demonstration purposes only; does not represent an actual result.
        prob = self._make_prob(
            num_nodes=2,
            aero_data=BWB_FREE_DATA,
            mach=[0.8, 0.82],
            alpha=[4.216, 3.146],
            altitude=[37500, 41000],
        )
        prob.run_model()

        expected_values = {
            Dynamic.Vehicle.LIFT_COEFFICIENT: (
                np.array([0.1276112, 0.05515637]),
                'unitless',
            ),
            Dynamic.Vehicle.DRAG_COEFFICIENT: (
                np.array([0.00599297, 0.00570541]),
                'unitless',
            ),
        }
        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tolerance=0.001)

        partial_data = prob.check_partials(method='cs', out_stream=None)
        assert_check_partials(partial_data, atol=9e-8, rtol=2e-7)


class TabularLowSpeedAeroTestCase(unittest.TestCase):
    """Tests for TabularLowSpeedAero against GASP takeoff/groundroll output."""

    # gear retraction start time at takeoff
    t_init_gear_to = 37.3
    # flap retraction start time at takeoff
    t_init_flaps_to = 47.6
    # takeoff flap deflection (deg)
    flap_defl_to = 10

    free_data = LARGE_SINGLE_AISLE_FREE_DATA
    flaps_data = get_aviary_resource_path(
        'models/large_single_aisle_1/aerodynamics_tables/large_single_aisle_1_aero_flaps.csv'
    )
    ground_data = get_aviary_resource_path(
        'models/large_single_aisle_1/aerodynamics_tables/large_single_aisle_1_aero_ground.csv'
    )

    def _make_prob(
        self,
        num_nodes,
        t_curr,
        altitude,
        mach,
        alpha,
        gross_mass,
        wing_area=1370.3,
        wing_span=117.8,
        wing_height=0.0,
        airport_alt=0.0,
    ):
        prob = om.Problem()
        prob.model = TabularLowSpeedAero(
            num_nodes=num_nodes,
            free_aero_data=self.free_data,
            flaps_aero_data=self.flaps_data,
            ground_aero_data=self.ground_data,
            connect_training_data=False,
            structured=True,
            extrapolate=True,
            retract_gear=True,
            retract_flaps=True,
        )
        prob.model.set_input_defaults(Aircraft.Wing.AREA, val=wing_area, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=wing_span, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.HEIGHT, val=wing_height, units='ft')
        prob.setup()

        prob.set_val('t_curr', t_curr, units='s')
        prob.set_val('airport_alt', airport_alt, units='ft')
        prob.set_val(Dynamic.Mission.ALTITUDE, altitude, units='ft')
        prob.set_val(Dynamic.Atmosphere.MACH, mach, units='unitless')
        prob.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, alpha, units='deg')
        prob.set_val(Aircraft.Design.GROSS_MASS, gross_mass, units='lbm')

        prob.set_val('flap_defl', self.flap_defl_to, units='deg')
        prob.set_val('t_init_gear', self.t_init_gear_to, units='s')
        prob.set_val('t_init_flaps', self.t_init_flaps_to, units='s')
        return prob

    def test_groundroll(self):
        # takeoff with flaps applied, gear down, zero alt
        prob = self._make_prob(
            num_nodes=4,
            t_curr=[0.0, 1.0, 2.0, 3.0],
            altitude=0,
            mach=[0.0, 0.009, 0.018, 0.026],
            alpha=0,
            gross_mass=175400.0,
        )
        # TODO set q if we want to test lift/drag forces
        prob.run_model()

        cl_exp = 0.5597 * np.ones(4)
        cd_exp = 0.0572 * np.ones(4)
        expected_values = {
            Dynamic.Vehicle.LIFT_COEFFICIENT: (cl_exp, 'unitless'),
            Dynamic.Vehicle.DRAG_COEFFICIENT: (cd_exp, 'unitless'),
        }
        # TODO yikes @ tolerances
        tolerances = {
            Dynamic.Vehicle.LIFT_COEFFICIENT: 0.1,
            Dynamic.Vehicle.DRAG_COEFFICIENT: 0.3,
        }
        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tolerance=tolerances[var_name])

        partial_data = prob.check_partials(
            method='fd', out_stream=None
        )  # fd because there is a cs in the time ramp
        assert_check_partials(partial_data, atol=3e-7, rtol=6e-5)

    def test_takeoff(self):
        # takeoff crossing flap retraction and gear retraction points
        prob = self._make_prob(
            num_nodes=8,
            t_curr=[37.0, 38.0, 39.0, 40.0, 47.0, 48.0, 49.0, 50.0],
            altitude=[44.2, 62.7, 84.6, 109.7, 373.0, 419.4, 465.3, 507.8],
            mach=[0.257, 0.260, 0.263, 0.265, 0.276, 0.277, 0.279, 0.280],
            alpha=[8.94, 8.74, 8.44, 8.24, 6.45, 6.34, 6.76, 7.59],
            gross_mass=175400.0,
        )
        # TODO set q if we want to test lift/drag forces
        prob.run_model()

        cl_exp = np.array([1.3734, 1.3489, 1.3179, 1.2979, 1.1356, 1.0645, 0.9573, 0.8876])
        cd_exp = np.array([0.1087, 0.1070, 0.1019, 0.0969, 0.0661, 0.0641, 0.0644, 0.0680])
        expected_values = {
            Dynamic.Vehicle.LIFT_COEFFICIENT: (cl_exp, 'unitless'),
            Dynamic.Vehicle.DRAG_COEFFICIENT: (cd_exp, 'unitless'),
        }
        # TODO yikes @ tolerances
        tolerances = {
            Dynamic.Vehicle.LIFT_COEFFICIENT: 0.02,
            Dynamic.Vehicle.DRAG_COEFFICIENT: 0.09,
        }
        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tolerance=tolerances[var_name])

        partial_data = prob.check_partials(
            method='fd', out_stream=None
        )  # fd because there is a cs in the time ramp
        # fd does very poorly with the t_curr, t_init, and duration values in the time ramp
        # because its step is so much bigger than cs. By decreasing the fd step size you
        # can see that the derivatives are right wrt these values.
        assert_check_partials(partial_data, atol=0.255, rtol=5e-7)


class GearDragIncrementTestCase(unittest.TestCase):
    """Tests for GearDragIncrement."""

    def test_case(self):
        prob = om.Problem()
        prob.model.add_subsystem(
            'drag_inc',
            GearDragIncrement(num_nodes=2),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        prob.setup(check=False, force_alloc_complex=True)
        prob.set_val(Aircraft.Design.GROSS_MASS, 175000, units='lbm')
        prob.set_val(Aircraft.Wing.AREA, 1000, units='ft**2')
        prob.set_val('flap_defl', [0.0, 0.3], units='deg')
        prob.run_model()

        assert_near_equal(prob['dCD'], [0.04308, 0.04296], 1e-4)
        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
