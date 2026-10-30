"""
Test file to test the outputs, derivatives, and IO of each sample component/group.
The name of this file needs to start with 'test' so that the testflo command will
find and run the file.
"""

import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.mission.energy_state.phases.simplified_takeoff import (
    FinalTakeoffConditions,
    StallSpeed,
    TakeoffGroup,
)
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


@use_tempdirs
class StallSpeedTest(unittest.TestCase):
    """Test computation in StallSpeed class."""

    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'comp',
            StallSpeed(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults('mass', val=181200.0, units='lbm')  # check
        self.prob.model.set_input_defaults(
            Dynamic.Atmosphere.DENSITY, val=1.225, units='kg/m**3'
        )  # check
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=1370.0, units='ft**2'
        )  # check (this is the reference wing area)
        self.prob.model.set_input_defaults('Cl_max', val=2.0000, units='unitless')  # check

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-5

        assert_near_equal(self.prob['v_stall'], 71.90002053, tol)  # not actual value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)  # check the partial derivatives

    def test_case_alt_gravity(self):
        self.prob.model_options['*'] = {Mission.GRAVITY: (10, 'm/s**2')}

        self.prob.setup(check=False, force_alloc_complex=True)
        self.prob.run_model()

        tol = 1e-5

        assert_near_equal(self.prob['v_stall'], 72.60535887, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)  # check the partial derivatives


@use_tempdirs
class FinalConditionsTest(unittest.TestCase):
    """Test final conditions computation in FinalTakeoffConditions class."""

    def setUp(self):
        self.prob = om.Problem()
        opts = {
            Mission.SEA_LEVEL_DENSITY: (0.0023769, 'slug/ft**3'),
        }
        self.prob.model.add_subsystem(
            'comp',
            FinalTakeoffConditions(**opts),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults('v_stall', val=100, units='m/s')  # not actual value
        self.prob.model.set_input_defaults('mass', val=181200.0, units='lbm')  # check
        self.prob.model.set_input_defaults(Mission.Takeoff.FUEL_MASS, val=577, units='lbm')  # check
        self.prob.model.set_input_defaults(
            Dynamic.Atmosphere.DENSITY,
            val=0.0023769,
            units='slug/ft**3',
        )  # check
        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, val=1370.0, units='ft**2')  # check
        self.prob.model.set_input_defaults(
            Mission.Takeoff.LIFT_COEFFICIENT_MAX, val=2.0000, units='unitless'
        )  # check
        self.prob.model.set_input_defaults(
            Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST, val=28928.0 * 2, units='lbf'
        )  # check
        self.prob.model.set_input_defaults(
            Mission.Takeoff.CLIMBOUT_THRUST_FRACTION, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Mission.Takeoff.LIFT_OVER_DRAG, val=17.354, units='unitless'
        )  # check

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-5

        expected_values = {
            # values not actual
            Mission.Takeoff.GROUND_DISTANCE: (6637.65417226, 'ft'),
            Mission.Takeoff.FINAL_VELOCITY: (123.09, 'm/s'),
            Mission.Takeoff.FINAL_MASS: (180623.0, 'lbm'),
            Mission.Takeoff.FINAL_ALTITUDE: (35, 'ft'),
        }

        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = self.prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tol)

    def test_case_alt_gravity(self):
        self.prob.model_options['*'] = {Mission.GRAVITY: (10, 'm/s**2')}

        self.prob.setup(check=False, force_alloc_complex=True)
        self.prob.run_model()
        tol = 1e-5

        expected_values = {
            Mission.Takeoff.GROUND_DISTANCE: (6867.55481846, 'ft'),
            Mission.Takeoff.FINAL_VELOCITY: (123.09, 'm/s'),
            Mission.Takeoff.FINAL_MASS: (180623.0, 'lbm'),
            Mission.Takeoff.FINAL_ALTITUDE: (35, 'ft'),
        }

        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = self.prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


@use_tempdirs
class TakeoffGroupTest(unittest.TestCase):
    """Test computation in TakeoffGroup."""

    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('group_example', TakeoffGroup(), promotes=['*'])

        self.prob.model.set_input_defaults(Mission.GROSS_MASS, val=181300.0, units='lbm')  # check
        self.prob.model.set_input_defaults(Mission.Taxi.FUEL_MASS_TAXI_OUT, val=101, units='lbm')
        self.prob.model.set_input_defaults(Mission.Takeoff.FUEL_MASS, val=577, units='lbm')  # check
        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, val=1370.0, units='ft**2')  # check
        self.prob.model.set_input_defaults(
            Mission.Takeoff.LIFT_COEFFICIENT_MAX, val=2.0000, units='unitless'
        )  # check
        self.prob.model.set_input_defaults(
            Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST, val=28928.0 * 2, units='lbf'
        )  # check
        self.prob.model.set_input_defaults(
            Mission.Takeoff.CLIMBOUT_THRUST_FRACTION, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Mission.Takeoff.LIFT_OVER_DRAG, val=17.354, units='unitless'
        )
        self.prob.model.set_input_defaults(Dynamic.Mission.ALTITUDE, val=0, units='ft')  # check
        self.prob.model.set_input_defaults(Dynamic.Mission.VELOCITY, 100, 'ft/s')

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-5

        expected_values = {
            'end_of_taxi_mass': (181199, 'lbm'),
            'v_stall': (71.90002053, 'm/s'),
            # values not actual
            Mission.Takeoff.GROUND_DISTANCE: (6637.65645404, 'ft'),
            Mission.Takeoff.FINAL_VELOCITY: (88.50175527, 'm/s'),
            Mission.Takeoff.FINAL_MASS: (180623.0, 'lbm'),
            Mission.Takeoff.FINAL_ALTITUDE: (35, 'ft'),
            Mission.Takeoff.FINAL_MACH: (0.26009873, 'unitless'),
        }

        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = self.prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tol)

        partial_data = self.prob.check_partials(
            out_stream=None, excludes=['*.standard_atmosphere'], method='cs'
        )
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
