import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.mission.gasp_based.ode.breguet_cruise_ode import BreguetCruiseODESolution, E_BreguetCruiseODESolution
from aviary.mission.gasp_based.ode.params import set_params_for_unit_tests
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.test_utils.default_subsystems import get_default_mission_subsystems
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Dynamic, Mission

from aviary.subsystems.propulsion.motor.motor_builder import MotorBuilder
from aviary.subsystems.propulsion.turboprop_model import TurbopropModel
from aviary.utils.preprocessors import preprocess_propulsion

class CruiseODETestCase(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

        aviary_options = get_option_defaults()
        aviary_options.set_val(Aircraft.Engine.GLOBAL_THROTTLE, True)
        default_mission_subsystems = get_default_mission_subsystems(
            'GASP', build_engine_deck(aviary_options)
        )

        self.prob.model = BreguetCruiseODESolution(
            num_nodes=2,
            aviary_options=aviary_options,
            core_subsystems=default_mission_subsystems,
        )

        self.prob.model.set_input_defaults(
            Dynamic.Atmosphere.MACH, np.array([0, 0]), units="unitless"
        )

    def test_cruise(self):
        # test partial derivatives
        self.prob.setup(check=False, force_alloc_complex=True)

        self.prob.set_val(Dynamic.Atmosphere.MACH, [0.7, 0.7], units="unitless")
        self.prob.set_val("interference_independent_of_shielded_area", 1.89927266)
        self.prob.set_val("drag_loss_due_to_shielded_wing_area", 68.02065834)

        set_params_for_unit_tests(self.prob)

        self.prob.run_model()

        tol = tol = 1e-6
        assert_near_equal(
            self.prob[Dynamic.Mission.VELOCITY_RATE], np.array([1.0, 1.0]), tol
        )
        assert_near_equal(
            self.prob[Dynamic.Mission.DISTANCE], np.array([0.0, 882.5769]), tol
        )
        assert_near_equal(self.prob["time"], np.array([0, 7913.69]), tol)
        assert_near_equal(
            self.prob[Dynamic.Mission.SPECIFIC_ENERGY_RATE_EXCESS],
            np.array([3.439203, 4.440962]),
            tol,
        )
        assert_near_equal(
            self.prob[Dynamic.Mission.ALTITUDE_RATE_MAX],
            np.array([-17.622456, -16.62070]),
            tol,
        )

        partial_data = self.prob.check_partials(
            out_stream=None, method="cs", excludes=["*USatm*", "*params*", "*aero*"]
        )
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class ElectricCruiseODETestCase(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

        aviary_options = get_option_defaults()
        aviary_options.set_val(Aircraft.Engine.GLOBAL_THROTTLE, True)
        aviary_options.set_val(
            Aircraft.Engine.DATA_FILE,
            'mission/gasp_based/ode/test/test_data/turbofan_23k_electrified.deck',
        )
        default_mission_subsystems = get_default_mission_subsystems(
            'GASP', build_engine_deck(aviary_options)
        )

        self.prob.model = E_BreguetCruiseODESolution(
            num_nodes=2,
            aviary_options=aviary_options,
            core_subsystems=default_mission_subsystems,
        )

        self.prob.model.set_input_defaults(
            Dynamic.Atmosphere.MACH, np.array([0, 0]), units="unitless"
        )

    def test_cruise(self):
        # test partial derivatives
        self.prob.setup(check=False, force_alloc_complex=True)

        self.prob.set_val(Dynamic.Atmosphere.MACH, [0.7, 0.7], units="unitless")
        self.prob.set_val("interference_independent_of_shielded_area", 1.89927266)
        self.prob.set_val("drag_loss_due_to_shielded_wing_area", 68.02065834)

        set_params_for_unit_tests(self.prob)

        self.prob.run_model()

        tol = tol = 1e-6
        assert_near_equal(
            self.prob[Dynamic.Mission.VELOCITY_RATE], np.array([1.0, 1.0]), tol
        )
        assert_near_equal(
            self.prob[Dynamic.Mission.DISTANCE], np.array([0.0, 66.66771412]), tol
        )
        assert_near_equal(self.prob["time"], np.array([0, 597.78110206]), tol)
        assert_near_equal(
            self.prob[Dynamic.Mission.SPECIFIC_ENERGY_RATE_EXCESS],
            np.array([3.439203, 4.440962]),
            tol,
        )
        assert_near_equal(
            self.prob[Dynamic.Mission.ALTITUDE_RATE_MAX],
            np.array([-17.622456, -16.62070]),
            tol,
        )
        assert_near_equal(
            self.prob[Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN_TOTAL],
            np.array([4.67455501, 4.33784647]),
            tol,
        )

        partial_data = self.prob.check_partials(
            out_stream=None, method="cs", excludes=["*USatm*", "*params*", "*aero*"]
        )
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class ElectricCruiseODETestCase2(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

        # add electrical motor
        motor_model = MotorBuilder()
        options = get_option_defaults()
        options.set_val(Aircraft.Engine.NUM_ENGINES, 2)
        options.set_val(Aircraft.Engine.SUBSONIC_FUEL_FLOW_SCALER, 1.0)
        options.set_val(Aircraft.Engine.SUPERSONIC_FUEL_FLOW_SCALER, 1.0)
        options.set_val(Aircraft.Engine.FUEL_FLOW_SCALER_CONSTANT_TERM, 0.0)
        options.set_val(Aircraft.Engine.FUEL_FLOW_SCALER_LINEAR_TERM, 1.0)
        options.set_val(Aircraft.Engine.CONSTANT_FUEL_CONSUMPTION, 0.0, units='lbm/h')
        options.set_val(Aircraft.Engine.SCALE_PERFORMANCE, True)
        options.set_val(Mission.Summary.FUEL_FLOW_SCALER, 1.0)
        options.set_val(Aircraft.Engine.SCALE_FACTOR, 1)
        options.set_val(Aircraft.Engine.GENERATE_FLIGHT_IDLE, False)
        options.set_val(Aircraft.Engine.IGNORE_NEGATIVE_THRUST, False)
        options.set_val(Aircraft.Engine.FLIGHT_IDLE_THRUST_FRACTION, 0.0)
        options.set_val(Aircraft.Engine.FLIGHT_IDLE_MAX_FRACTION, 1.0)
        options.set_val(Aircraft.Engine.FLIGHT_IDLE_MIN_FRACTION, 0.08)
        options.set_val(Aircraft.Engine.GEOPOTENTIAL_ALT, False)
        options.set_val(Aircraft.Engine.INTERPOLATION_METHOD, 'slinear')
        options.set_val(Aircraft.Engine.FIXED_RPM, 1455.13090827, units='rpm')
        options.set_val(Aircraft.Engine.Propeller.COMPUTE_INSTALLATION_LOSS, val=True)
        options.set_val(Aircraft.Engine.Propeller.NUM_BLADES, val=4, units='unitless')

        engine = TurbopropModel(
            options=options, shaft_power_model=motor_model, propeller_model=None
        )
        preprocess_propulsion(options, [engine])

        aviary_options = get_option_defaults()
        aviary_options.set_val(Aircraft.Engine.GLOBAL_THROTTLE, True)
        default_mission_subsystems = get_default_mission_subsystems(
            'GASP', engine
        )

        self.prob.model = E_BreguetCruiseODESolution(
            num_nodes=2,
            aviary_options=aviary_options,
            core_subsystems=default_mission_subsystems,
        )

        self.prob.model.set_input_defaults(
            Dynamic.Atmosphere.MACH, 0 * np.ones(2), units="unitless"
        )

    def test_cruise(self):
        # test partial derivatives
        self.prob.setup(check=False, force_alloc_complex=True)

        self.prob.set_val(Dynamic.Vehicle.Propulsion.RPM,
                          np.ones(2) * 2000.0, units='rpm')
        self.prob.set_val(Aircraft.Engine.Propeller.DIAMETER, 10.5, units="ft")
        self.prob.set_val(
            Aircraft.Engine.Propeller.ACTIVITY_FACTOR, 114.0, units="unitless"
        )
        self.prob.set_val(
            Aircraft.Engine.Propeller.INTEGRATED_LIFT_COEFFICIENT, 0.5, units="unitless"
        )
        self.prob.set_val(Aircraft.Engine.Propeller.TIP_SPEED_MAX, 800, units="ft/s")

        self.prob.set_val(Dynamic.Atmosphere.SPEED_OF_SOUND, 344 * np.ones(2), units="ft/s")
        self.prob.set_val(Dynamic.Atmosphere.DENSITY, 1.05 * np.ones(2), units="kg/m**3")
        #self.prob.set_val("interference_independent_of_shielded_area", 1.89927266)
        #self.prob.set_val("drag_loss_due_to_shielded_wing_area", 68.02065834)
        #self.prob.set_val(Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN,
        #                  10079.422 * np.ones(2), units="kW")
        self.prob.set_val(Dynamic.Atmosphere.MACH, 0.7 * np.ones(2), units="unitless")

        set_params_for_unit_tests(self.prob)

        self.prob.run_model()

        print(f"test: {self.prob[Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN_TOTAL]}")
        print(f"test: {self.prob[Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL]}")
        print(f"test: {self.prob[Dynamic.Mission.VELOCITY_RATE]}")
        print(f"test: {self.prob[Dynamic.Mission.DISTANCE]}")

        tol = tol = 1e-6
        assert_near_equal(
            self.prob[Dynamic.Mission.VELOCITY_RATE], np.array([1.0, 1.0]), tol
        )
        assert_near_equal(
            self.prob[Dynamic.Mission.DISTANCE], np.array([0.0, 882.5769]), tol
        )
        assert_near_equal(self.prob["time"], np.array([0, 7913.69]), tol)
        assert_near_equal(
            self.prob[Dynamic.Mission.SPECIFIC_ENERGY_RATE_EXCESS],
            np.array([3.439203, 4.440962]),
            tol,
        )
        assert_near_equal(
            self.prob[Dynamic.Mission.ALTITUDE_RATE_MAX],
            np.array([-17.622456, -16.62070]),
            tol,
        )

        partial_data = self.prob.check_partials(
            out_stream=None, method="cs", excludes=["*USatm*", "*params*", "*aero*"]
        )
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


if __name__ == "__main__":
    # unittest.main()
    test = ElectricCruiseODETestCase()
    test.setUp()
    test.test_cruise()
