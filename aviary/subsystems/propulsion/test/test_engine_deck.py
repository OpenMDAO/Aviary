import csv
import unittest
from pathlib import Path

from openmdao.utils.assert_utils import assert_near_equal

from aviary.subsystems.propulsion.engine_deck import EngineDeck
from aviary.subsystems.propulsion.utils import EngineModelVariables as keys
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.named_values import NamedValues
from aviary.validation_cases.validation_tests import get_flops_inputs
from aviary.variable_info.variables import Aircraft


class EngineDeckTest(unittest.TestCase):
    def test_flight_idle(self):
        # original test data was created with old version of converted GASP engine deck w/o
        # rounding, so tol must be lower here for comparison with modern engine
        tol = 1e-4

        aviary_values = get_flops_inputs('LargeSingleAisle2FLOPS')
        # Test data grabbed from LEAPS uses the global throttle approach
        aviary_values.set_val(Aircraft.Engine.GLOBAL_THROTTLE, True)

        model = build_engine_deck(aviary_values)

        expected_mach_number = []
        expected_altitude = []
        expected_throttle = []
        expected_thrust = []
        expected_fuel_flow_rate = []

        # hardcoded data of processed engine model from LEAPS1 after flight idle
        # point generation, sorted in Aviary order
        with open(Path(__file__).parents[0] / 'engine_model_test_data_turbofan_24k_1.csv') as file:
            reader = csv.reader(file)
            for row in reader:
                expected_mach_number.append(float(row[0]))
                expected_altitude.append(float(row[1]))
                expected_throttle.append(float(row[2]))
                expected_thrust.append(float(row[3]))
                expected_fuel_flow_rate.append(float(row[4]))

        mach_number = model.data[keys.MACH]
        altitude = model.data[keys.ALTITUDE]
        throttle = model.data[keys.THROTTLE]
        thrust = model.data[keys.THRUST]
        fuel_flow_rate = model.data[keys.FUEL_FLOW]

        assert_near_equal(mach_number, expected_mach_number, tolerance=tol)
        assert_near_equal(altitude, expected_altitude, tolerance=tol)
        assert_near_equal(throttle, expected_throttle, tolerance=tol)
        assert_near_equal(thrust, expected_thrust, tolerance=tol)
        assert_near_equal(fuel_flow_rate, expected_fuel_flow_rate, tolerance=tol)
        # no need for check_partials

    def test_flight_idle_2(self):
        tol = 1e-6

        aviary_values = get_flops_inputs('LargeSingleAisle1FLOPS')

        model = build_engine_deck(aviary_values)

        # hardcoded data of processed engine model from LEAPS1 after flight idle
        # point generation, sorted in Aviary order

        expected_mach_number = []
        expected_altitude = []
        expected_throttle = []
        expected_thrust = []
        expected_fuel_flow_rate = []

        # hardcoded data of processed engine model from LEAPS1 after flight idle
        # point generation, sorted in Aviary order
        with open(Path(__file__).parents[0] / 'engine_model_test_data_turbofan_28k.csv') as file:
            reader = csv.reader(file)
            for row in reader:
                expected_mach_number.append(float(row[0]))
                expected_altitude.append(float(row[1]))
                expected_throttle.append(float(row[2]))
                expected_thrust.append(float(row[3]))
                expected_fuel_flow_rate.append(float(row[4]))

        mach_number = model.data[keys.MACH]
        altitude = model.data[keys.ALTITUDE]
        # throttle = model.data[keys.THROTTLE]
        thrust = model.data[keys.THRUST]
        fuel_flow_rate = model.data[keys.FUEL_FLOW]

        assert_near_equal(mach_number, expected_mach_number, tolerance=tol)
        assert_near_equal(altitude, expected_altitude, tolerance=tol)
        # assert_near_equal(throttle, expected_throttle, tolerance=tol)
        assert_near_equal(thrust, expected_thrust, tolerance=tol)
        assert_near_equal(fuel_flow_rate, expected_fuel_flow_rate, tolerance=tol)

    def test_load_from_memory(self):
        tol = 1e-6

        aviary_values = get_flops_inputs('LargeSingleAisle2FLOPS')

        expected_mach_number = []
        expected_altitude = []
        expected_throttle = []
        expected_thrust = []
        expected_fuel_flow_rate = []

        # hardcoded data of processed engine model from LEAPS1 after flight idle
        # point generation, sorted in Aviary order
        with open(Path(__file__).parents[0] / 'engine_model_test_data_turbofan_24k_1.csv') as file:
            reader = csv.reader(file)
            for row in reader:
                expected_mach_number.append(float(row[0]))
                expected_altitude.append(float(row[1]))
                expected_throttle.append(float(row[2]))
                expected_thrust.append(float(row[3]))
                expected_fuel_flow_rate.append(float(row[4]))

        data_input = NamedValues()
        data_input.set_val('mach', expected_mach_number, 'unitless')
        data_input.set_val('altitude', expected_altitude, 'ft')
        data_input.set_val('throttle', expected_throttle, 'unitless')
        data_input.set_val('thrust', expected_thrust, 'lbf')
        data_input.set_val('fuel_flow', expected_fuel_flow_rate, 'lbm/h')

        model = EngineDeck('engine', aviary_values, data_input)

        mach_number = model.data[keys.MACH]
        altitude = model.data[keys.ALTITUDE]
        # throttle = model.data[keys.THROTTLE]
        thrust = model.data[keys.THRUST]
        fuel_flow_rate = model.data[keys.FUEL_FLOW]

        assert_near_equal(mach_number, expected_mach_number, tolerance=tol)
        assert_near_equal(altitude, expected_altitude, tolerance=tol)
        # assert_near_equal(throttle, expected_throttle, tolerance=tol)
        assert_near_equal(thrust, expected_thrust, tolerance=tol)
        assert_near_equal(fuel_flow_rate, expected_fuel_flow_rate, tolerance=tol)


if __name__ == '__main__':
    unittest.main()
