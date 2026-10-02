import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.mission.two_dof.ode.takeoff_ode import RequiredLift, TakeOffODE
from aviary.mission.two_dof.ode.test.params import set_params_for_unit_tests
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.test_utils.default_subsystems import get_default_mission_subsystems
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


@use_tempdirs
class GroundrollODETestCase(unittest.TestCase):
    """Test groundroll ODE."""

    def setUp(self):
        self.prob = om.Problem()

        aviary_options = get_option_defaults()
        aviary_options.set_val(Aircraft.Engine.GLOBAL_THROTTLE, True)
        # aviary_options.set_val(Mission.GRAVITY, val=32.2, units='ft/s**2')
        default_mission_subsystems = get_default_mission_subsystems(
            'GASP', [build_engine_deck(aviary_options)]
        )

        self.prob.model = TakeOffODE(
            num_nodes=2,
            ground_roll=True,
            aviary_options=aviary_options,
            subsystems=default_mission_subsystems,
        )

        setup_model_options(self.prob, aviary_options)

    def test_groundroll(self):
        self.prob.setup(check=False, force_alloc_complex=True)

        set_params_for_unit_tests(self.prob)

        self.prob.set_val('t_curr', [1, 2], units='s')
        self.prob.set_val('aircraft:wing:incidence', 0, units='deg')
        self.prob.set_val('interference_independent_of_shielded_area', 1.89927266)
        self.prob.set_val('drag_loss_due_to_shielded_wing_area', 68.02065834)
        self.prob.set_val(Aircraft.Wing.FORM_FACTOR, 1.25)
        self.prob.set_val(Aircraft.VerticalTail.FORM_FACTOR, 1.25)
        self.prob.set_val(Aircraft.HorizontalTail.FORM_FACTOR, 1.25)
        self.prob.set_val(Aircraft.Fuselage.FORM_FACTOR, 1.05557953)
        self.prob.set_val(Dynamic.Mission.VELOCITY, [75, 150], units='kn')
        self.prob.set_val(Dynamic.Vehicle.MASS, [100000, 100000], units='lbm')
        self.prob.set_val(Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT, 0.02)

        self.prob.run_model()

        tol = 1e-6
        expected_values = {
            Dynamic.Mission.VELOCITY_RATE: ([14.5712877, 11.8647389], 'ft/s**2'),
            Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE: ([0.0, 0.0], 'rad/s'),
            Dynamic.Mission.ALTITUDE_RATE: ([0.0, 0.0], 'ft/s'),
            Dynamic.Mission.DISTANCE_RATE: ([126.58573928, 253.17147857], 'ft/s'),
            'normal_force': ([85313.25425063, 41138.11842255], 'lbf'),
            'fuselage_pitch': ([0.0, 0.0], 'deg'),
            'dmass_dv': ([-0.48559186, -0.60946082], 'lbm/knot'),
        }

        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = self.prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tol)

        with self.subTest(check='partials'):
            partial_data = self.prob.check_partials(
                out_stream=None, method='cs', excludes=['*params*', '*aero*']
            )
            assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class RotationODETestCase(unittest.TestCase):
    """Test 2-degrees-of-freedom rotation ODE."""

    def setUp(self):
        self.prob = om.Problem()

        aviary_options = get_option_defaults()
        aviary_options.set_val(Aircraft.Engine.GLOBAL_THROTTLE, True)
        # aviary_options.set_val(Mission.GRAVITY, val=32.2, units='ft/s**2')
        default_mission_subsystems = get_default_mission_subsystems(
            'GASP', [build_engine_deck(aviary_options)]
        )

        self.prob.model = TakeOffODE(
            num_nodes=2,
            rotation=True,
            aviary_options=aviary_options,
            subsystems=default_mission_subsystems,
        )
        setup_model_options(self.prob, aviary_options)

    def test_rotation(self):
        self.prob.setup(check=False, force_alloc_complex=True)

        self.prob.set_val(Aircraft.Wing.INCIDENCE, 1.5, units='deg')
        self.prob.set_val(Dynamic.Vehicle.MASS, [100000, 100000], units='lbm')
        self.prob.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, [1.5, 1.5], units='deg')
        self.prob.set_val(Dynamic.Mission.VELOCITY, [100, 100], units='kn')
        self.prob.set_val('t_curr', [1, 2], units='s')
        self.prob.set_val('interference_independent_of_shielded_area', 1.89927266)
        self.prob.set_val('drag_loss_due_to_shielded_wing_area', 68.02065834)
        self.prob.set_val(Aircraft.Wing.FORM_FACTOR, 1.25)
        self.prob.set_val(Aircraft.VerticalTail.FORM_FACTOR, 1.25)
        self.prob.set_val(Aircraft.HorizontalTail.FORM_FACTOR, 1.25)
        self.prob.set_val(Aircraft.Fuselage.FORM_FACTOR, 1.05557953)

        set_params_for_unit_tests(self.prob)

        self.prob.run_model()

        tol = 1e-6
        expected_values = {
            Dynamic.Mission.VELOCITY_RATE: ([13.67772614, 13.67772614], 'ft/s**2'),
            Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE: ([0.0, 0.0], 'rad/s'),
            Dynamic.Mission.ALTITUDE_RATE: ([0.0, 0.0], 'ft/s'),
            Dynamic.Mission.DISTANCE_RATE: ([168.781, 168.781], 'ft/s'),
            'normal_force': ([66936.59676831, 66936.59676831], 'lbf'),
            'fuselage_pitch': ([0.0, 0.0], 'deg'),
        }

        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = self.prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tol)

        with self.subTest(check='partials'):
            partial_data = self.prob.check_partials(
                out_stream=None, method='cs', excludes=['*params*', '*aero*']
            )
            assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


@use_tempdirs
class AscentODETestCase(unittest.TestCase):
    """Test 2-degrees-of-freedom ascent ODE."""

    def setUp(self):
        self.prob = om.Problem()

        aviary_options = get_option_defaults()
        aviary_options.set_val(Aircraft.Engine.GLOBAL_THROTTLE, True)
        # aviary_options.set_val(Mission.GRAVITY, val=32.2, units='ft/s**2')
        aviary_options.set_val(Aircraft.Engine.NUM_ENGINES, val=[2], units='unitless')
        default_mission_subsystems = get_default_mission_subsystems(
            'GASP', [build_engine_deck(aviary_options)]
        )

        self.prob.model = TakeOffODE(
            num_nodes=2, aviary_options=aviary_options, subsystems=default_mission_subsystems
        )

        setup_model_options(self.prob, AviaryValues(aviary_options))

    def test_ascent(self):
        # Test partial derivatives
        self.prob.setup(check=False, force_alloc_complex=True)

        # TODO: These values are kind of hokey, but were in the previous test.
        self.prob.set_val(Dynamic.Mission.VELOCITY, [100, 100], units='kn')
        self.prob.set_val(Dynamic.Vehicle.MASS, [1, 1], units='kg')

        self.prob.set_val('t_curr', [1, 2], units='s')
        self.prob.set_val('interference_independent_of_shielded_area', 1.89927266)
        self.prob.set_val('drag_loss_due_to_shielded_wing_area', 68.02065834)
        self.prob.set_val(Aircraft.Wing.FORM_FACTOR, 1.25)
        self.prob.set_val(Aircraft.VerticalTail.FORM_FACTOR, 1.25)
        self.prob.set_val(Aircraft.HorizontalTail.FORM_FACTOR, 1.25)

        set_params_for_unit_tests(self.prob)

        self.prob.run_model()

        tol = 1e-6
        expected_values = {
            Dynamic.Mission.VELOCITY_RATE: ([641639.45047776, 641639.45047776], 'ft/s**2'),
            Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE: ([2258.55674809, 2258.55674809], 'rad/s'),
            Dynamic.Mission.ALTITUDE_RATE: ([0.0, 0.0], 'ft/s'),
            Dynamic.Mission.DISTANCE_RATE: ([168.781, 168.781], 'ft/s'),
            'angle_of_attack_rate': ([0.0, 0.0], 'deg/s'),
            'normal_force': ([0.0, 0.0], 'lbf'),
            'fuselage_pitch': ([0.0, 0.0], 'deg'),
            'load_factor': ([11849.10281268, 11849.10281268], 'unitless'),
        }

        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = self.prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tol)

        with self.subTest(check='partials'):
            partial_data = self.prob.check_partials(
                out_stream=None, method='cs', excludes=['*params*', '*aero*']
            )
            assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class RequiredLiftTestCase(unittest.TestCase):
    """Test the RequiredLift component used by AlphaModes.REQUIRED_LIFT."""

    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('required_lift', RequiredLift(num_nodes=2), promotes=['*'])

        aviary_options = get_option_defaults()
        setup_model_options(self.prob, aviary_options)

    def test_required_lift(self):
        self.prob.setup(check=False, force_alloc_complex=True)

        # mass and wing incidence borrowed from RotationODETestCase
        self.prob.set_val(Dynamic.Vehicle.MASS, [100000, 100000], units='lbm')
        self.prob.set_val(Aircraft.Wing.INCIDENCE, 1.5, units='deg')
        self.prob.set_val(Dynamic.Vehicle.Propulsion.THRUST_TOTAL, [20000, 18000], units='lbf')
        self.prob.set_val(Dynamic.Mission.FLIGHT_PATH_ANGLE, [5, 7], units='deg')
        self.prob.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, [8, 6], units='deg')

        self.prob.run_model()

        tol = 1e-6
        expected_values = {
            'required_lift': ([96913.46751237, 96965.82140898], 'lbf'),
        }

        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = self.prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tol)

        with self.subTest(check='partials'):
            partial_data = self.prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


if __name__ == '__main__':
    unittest.main()
