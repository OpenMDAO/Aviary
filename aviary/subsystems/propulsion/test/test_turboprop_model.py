import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.subsystems.atmosphere.atmosphere import Atmosphere
from aviary.subsystems.propulsion.motor.motor_builder import MotorBuilder
from aviary.subsystems.propulsion.propeller.propeller_performance import PropellerPerformance
from aviary.subsystems.propulsion.turboprop_model import TurbopropModel
from aviary.subsystems.subsystem_builder import SubsystemBuilder
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.functions import get_path
from aviary.utils.preprocessors import preprocess_propulsion
from aviary.variable_info.enums import SpeedType
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


@use_tempdirs
class TurbopropMissionTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    def prepare_model(
        self,
        options: AviaryValues,
        test_points=[(0, 0, 0), (0, 0, 1)],
        shp_model=None,
        prop_model=None,
        **kwargs,
    ):
        options.set_val(Aircraft.Engine.NUM_ENGINES, 2)
        options.set_val(Aircraft.Engine.SUBSONIC_FUEL_FLOW_SCALER, 1.0)
        options.set_val(Aircraft.Engine.SUPERSONIC_FUEL_FLOW_SCALER, 1.0)
        options.set_val(Aircraft.Engine.FUEL_FLOW_SCALER_CONSTANT_TERM, 0.0)
        options.set_val(Aircraft.Engine.FUEL_FLOW_SCALER_LINEAR_TERM, 1.0)
        options.set_val(Aircraft.Engine.CONSTANT_FUEL_CONSUMPTION, 0.0, units='lbm/h')
        options.set_val(Aircraft.Engine.SCALE_FACTOR, 1)
        options.set_val(Aircraft.Engine.GENERATE_FLIGHT_IDLE, False)
        options.set_val(Aircraft.Engine.IGNORE_NEGATIVE_THRUST, False)
        options.set_val(Aircraft.Engine.FLIGHT_IDLE_THRUST_FRACTION, 0.0)
        options.set_val(Aircraft.Engine.FLIGHT_IDLE_MAX_FRACTION, 1.0)
        options.set_val(Aircraft.Engine.FLIGHT_IDLE_MIN_FRACTION, 0.08)
        options.set_val(Aircraft.Engine.GEOPOTENTIAL_ALT, False)
        options.set_val(Aircraft.Engine.INTERPOLATION_METHOD, 'slinear')
        options.set_val(
            Aircraft.Engine.Propeller.COMPUTE_INSTALLATION_LOSS,
            val=True,
            units='unitless',
        )
        options.set_val(Aircraft.Engine.Propeller.NUM_BLADES, val=4, units='unitless')

        num_nodes = len(test_points)

        engine = TurbopropModel(
            options=options, shaft_power_model=shp_model, propeller_model=prop_model
        )

        preprocess_propulsion(options, [engine])

        machs, alts, throttles = zip(*test_points)
        IVC = om.IndepVarComp(Dynamic.Atmosphere.MACH, np.array(machs), units='unitless')
        IVC.add_output(Dynamic.Mission.ALTITUDE, np.array(alts), units='ft')
        IVC.add_output(Dynamic.Vehicle.Propulsion.THROTTLE, np.array(throttles), units='unitless')
        self.prob.model.add_subsystem('IVC', IVC, promotes=['*'])

        # calculate atmospheric properties
        self.prob.model.add_subsystem(
            name='atmosphere',
            subsys=Atmosphere(num_nodes=num_nodes, input_speed_type=SpeedType.MACH),
            promotes=['*'],
        )

        # Put it all in an outer group called "propulsion" so that model structure looks like
        # aviary for model options.
        propulsion_group = self.prob.model.add_subsystem('propulsion', om.Group(), promotes=['*'])
        propulsion_group.add_subsystem(
            engine.name,
            subsys=engine.build_mission(
                num_nodes=num_nodes,
                aviary_inputs=options,
                user_options={},
                subsystem_options=kwargs,
            ),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        setup_model_options(self.prob, options, engine_models=[engine])

        self.prob.setup(force_alloc_complex=False)
        self.prob.set_val(Aircraft.Engine.SCALE_FACTOR, 1, units='unitless')

    def get_results(self, point_names=None, display_results=False):
        shp = self.prob.get_val(Dynamic.Vehicle.Propulsion.SHAFT_POWER, units='hp')
        total_thrust = self.prob.get_val(Dynamic.Vehicle.Propulsion.THRUST, units='lbf')
        prop_thrust = self.prob.get_val('thrust_summation.propeller_thrust', units='lbf')
        tailpipe_thrust = self.prob.get_val('thrust_summation.turboshaft_thrust', units='lbf')
        # max_thrust = self.prob.get_val(Dynamic.Vehicle.Propulsion.THRUST_MAX, units='lbf')
        fuel_flow = self.prob.get_val(
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE, units='lbm/h'
        )

        results = []
        for n, _ in enumerate(shp):
            results.append(
                (
                    shp[n],
                    tailpipe_thrust[n],
                    prop_thrust[n],
                    total_thrust[n],
                    # max_thrust[n],
                    fuel_flow[n],
                )
            )
        return results

    def test_case_1(self):
        # test case using GASP-derived engine deck and "user specified" prop model
        filename = get_path('models/engines/turboshaft_1120hp.csv')
        # Mach, alt, throttle @ idle, SLS, TOC
        test_points = [(0, 0, 0), (0, 0, 1), (0.6, 25000, 1)]
        # shp, tailpipe thrust, prop_thrust, total_thrust, max_thrust, fuel flow
        truth_vals = [
            (
                111.99960961,
                37.7,
                610.28630998,
                647.98630998,
                # 4183.87495338,
                -195.8,
            ),
            (
                1119.99609612,
                136.3,
                4047.57495338,
                4183.87495338,
                # 4183.87495338,
                -644.0,
            ),
            (
                778.21130479,
                21.3,
                558.33650216,
                579.63650216,
                # 579.63650216,
                -839.7,
            ),
        ]

        options = get_option_defaults()
        options.set_val(
            Aircraft.Engine.Propeller.COMPUTE_INSTALLATION_LOSS,
            val=True,
            units='unitless',
        )
        options.set_val(
            Aircraft.Engine.FIXED_RPM,
            1455.13090827,
            units='rpm',
        )
        options.set_val(Aircraft.Engine.Propeller.NUM_BLADES, val=4, units='unitless')
        options.set_val('speed_type', SpeedType.MACH)

        prop_group = ExamplePropModel('custom_prop_model')

        options.set_val(Aircraft.Engine.DATA_FILE, filename)

        self.prepare_model(options, test_points, prop_model=prop_group)

        self.prob.set_val(Aircraft.Engine.Propeller.DIAMETER, 10.5, units='ft')
        self.prob.set_val(Aircraft.Engine.Propeller.ACTIVITY_FACTOR, 114.0, units='unitless')
        # self.prob.set_val(Dynamic.Mission.PERCENT_ROTOR_RPM_CORRECTED,
        #                   np.array([1, 1, 0.7]), units='unitless')
        self.prob.set_val(
            Aircraft.Engine.Propeller.INTEGRATED_LIFT_COEFFICIENT, 0.5, units='unitless'
        )

        self.prob.set_val(Aircraft.Engine.Propeller.TIP_SPEED_MAX, 800, units='ft/s')

        self.prob.run_model()
        results = self.get_results()

        expected_values = {
            'point_0 (idle)': truth_vals[0],
            'point_1 (SLS)': truth_vals[1],
            'point_2 (TOC)': truth_vals[2],
        }

        for point_name, expected in expected_values.items():
            with self.subTest(var=point_name):
                idx = list(expected_values.keys()).index(point_name)
                assert_near_equal(results[idx], expected, tolerance=1.5e-10)

        # because Hamilton Standard model uses fd method, the following may not be accurate.
        partial_data = self.prob.check_partials(out_stream=None, form='central')
        assert_check_partials(partial_data, atol=0.2, rtol=0.2)

    def test_case_2(self):
        # test case using GASP-derived engine deck and default HS prop model.
        filename = get_path('models/engines/turboshaft_1120hp.csv')
        test_points = [(0.001, 0, 0), (0, 0, 1), (0.6, 25000, 1)]
        truth_vals = [
            (
                111.99507922,
                37.507376,
                610.67122085,
                648.17859685,
                # 4174.43077943,
                -195.78762,
            ),
            (
                1119.99609612,
                136.3,
                4047.57495338,
                4183.87495338,
                # 4183.87495338,
                -644.0,
            ),
            (
                778.21130479,
                21.3,
                558.33650216,
                579.63650216,
                # 579.63650216,
                -839.7,
            ),
        ]

        options = get_option_defaults()
        options.set_val(Aircraft.Engine.DATA_FILE, filename)
        options.set_val(
            Aircraft.Engine.FIXED_RPM,
            1455.13090827,
            units='rpm',
        )
        self.prepare_model(options, test_points)

        self.prob.set_val(Aircraft.Engine.Propeller.DIAMETER, 10.5, units='ft')
        self.prob.set_val(Aircraft.Engine.Propeller.ACTIVITY_FACTOR, 114.0, units='unitless')
        self.prob.set_val(
            Aircraft.Engine.Propeller.INTEGRATED_LIFT_COEFFICIENT, 0.5, units='unitless'
        )

        self.prob.set_val(Aircraft.Engine.Propeller.TIP_SPEED_MAX, 800, units='ft/s')

        self.prob.run_model()

        results = self.get_results()

        expected_values = {
            'point_0 (M=0.001, alt=0, idle)': truth_vals[0],
            'point_1 (M=0, alt=0, SLS)': truth_vals[1],
            'point_2 (M=0.6, alt=25k, TOC)': truth_vals[2],
        }

        for point_name, expected in expected_values.items():
            with self.subTest(var=point_name):
                idx = list(expected_values.keys()).index(point_name)
                assert_near_equal(results[idx], expected, tolerance=1.5e-10)

        partial_data = self.prob.check_partials(out_stream=None, form='central')
        assert_check_partials(partial_data, atol=0.15, rtol=0.15)

    def test_case_3(self):
        # test case using GASP-derived engine deck w/o tailpipe thrust and default
        # HS prop model.
        filename = get_path('models/engines/turboshaft_1120hp_no_tailpipe.csv')
        test_points = [(0, 0, 0), (0, 0, 1), (0.6, 25000, 1)]
        truth_vals = [
            (
                111.99960961,
                0.0,
                610.28630998,
                610.28630998,
                # 4047.57495338,
                -195.8,
            ),
            (
                1119.99609612,
                0.0,
                4047.57495338,
                4047.57495338,
                # 4047.57495338,
                -644.0,
            ),
            (
                778.21130479,
                0.0,
                558.33650216,
                558.33650216,
                # 558.33650216,
                -839.7,
            ),
        ]

        options = get_option_defaults()
        options.set_val(Aircraft.Engine.DATA_FILE, filename)
        options.set_val(
            Aircraft.Engine.FIXED_RPM,
            1455.13090827,
            units='rpm',
        )
        self.prepare_model(options, test_points)

        self.prob.set_val(Aircraft.Engine.Propeller.DIAMETER, 10.5, units='ft')
        self.prob.set_val(Aircraft.Engine.Propeller.ACTIVITY_FACTOR, 114.0, units='unitless')
        self.prob.set_val(
            Aircraft.Engine.Propeller.INTEGRATED_LIFT_COEFFICIENT, 0.5, units='unitless'
        )

        self.prob.set_val(Aircraft.Engine.Propeller.TIP_SPEED_MAX, 800, units='ft/s')

        self.prob.run_model()

        results = self.get_results()

        expected_values = {
            'point_0 (idle)': truth_vals[0],
            'point_1 (SLS)': truth_vals[1],
            'point_2 (TOC)': truth_vals[2],
        }

        for point_name, expected in expected_values.items():
            with self.subTest(var=point_name):
                idx = list(expected_values.keys()).index(point_name)
                assert_near_equal(results[idx], expected, tolerance=1.5e-10)

        # NOTE: There isn't much point in checking the partials of a component that computes them with FD.
        partial_data = self.prob.check_partials(out_stream=None, form='forward', step=1.01e-6)
        assert_check_partials(partial_data, atol=1e10, rtol=1e-3)

    def test_electroprop_fixed_RPM(self):
        # test case using electric motor and default HS prop model and fixed RPM.
        test_points = [(0, 0, 0), (0, 0, 1), (0.6, 25000, 1)]

        options = get_option_defaults()

        shp_file = get_path('electric_motor_1800Nm_6000rpm.csv')
        options.set_val(Aircraft.Engine.Motor.DATA_FILE, shp_file)
        options.set_val(Aircraft.Engine.RPM_DESIGN, 6000, 'rpm')
        options.set_val(
            Aircraft.Engine.FIXED_RPM,
            1455.13090827,
            units='rpm',
        )

        self.prepare_model(options, test_points, shp_model=MotorBuilder(), input_rpm=True)

        self.prob.set_val(Aircraft.Engine.Propeller.DIAMETER, 10.5, units='ft')
        self.prob.set_val(Aircraft.Engine.Propeller.ACTIVITY_FACTOR, 114.0, units='unitless')
        self.prob.set_val(
            Aircraft.Engine.Propeller.INTEGRATED_LIFT_COEFFICIENT, 0.5, units='unitless'
        )

        self.prob.set_val(Aircraft.Engine.Propeller.TIP_SPEED_MAX, 800, units='ft/s')

        self.prob.run_model()

        shp_expected = [0.0, 367.82313837, 367.82313837]
        prop_thrust_expected = total_thrust_expected = [
            610.28631174,
            2083.18866404,
            184.42047241,
        ]
        electric_power_expected = [0.0, 303.31014553, 303.31014553]

        expected_values = {
            Dynamic.Vehicle.Propulsion.SHAFT_POWER: (shp_expected, 'hp', 1e-8),
            Dynamic.Vehicle.Propulsion.THRUST: (total_thrust_expected, 'lbf', 1e-8),
            'thrust_summation.propeller_thrust': (prop_thrust_expected, 'lbf', 1e-8),
            Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN: (electric_power_expected, 'kW', 2e-7),
        }

        for var_name, (expected, units, tol) in expected_values.items():
            with self.subTest(var=var_name):
                actual = self.prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tolerance=tol)

        # NOTE: There isn't much point in checking the partials of a component that computes them
        # with FD.
        partial_data = self.prob.check_partials(out_stream=None, form='forward', step=1.01e-6)
        assert_check_partials(partial_data, atol=1e10, rtol=1e-3)

    def test_electroprop_calc_RPM(self):
        # test case using electric motor and default HS prop model and RPM that scales with throttle.
        test_points = [(0, 0, 0.01), (0, 0, 0.5), (0, 0, 1)]

        options = get_option_defaults()

        shp_file = get_path('electric_motor_1800Nm_6000rpm.csv')
        options.set_val(Aircraft.Engine.Motor.DATA_FILE, shp_file)
        options.set_val(Aircraft.Engine.RPM_DESIGN, 6000, 'rpm')
        options.delete(Aircraft.Engine.FIXED_RPM)

        self.prepare_model(options, test_points, shp_model=MotorBuilder(), input_rpm=True)

        self.prob.set_val(Aircraft.Engine.Propeller.DIAMETER, 10.5, units='ft')
        self.prob.set_val(Aircraft.Engine.Propeller.ACTIVITY_FACTOR, 114.0, units='unitless')
        self.prob.set_val(
            Aircraft.Engine.Propeller.INTEGRATED_LIFT_COEFFICIENT, 0.5, units='unitless'
        )

        self.prob.set_val(Aircraft.Engine.Propeller.TIP_SPEED_MAX, 800, units='ft/s')

        self.prob.run_model()

        shp_expected = [1.51665999, 536.2202822, 1516.659991]
        prop_thrust_expected = total_thrust_expected = [
            103.76048662,
            5188.02430387,
            10376.04860512,
        ]
        electric_power_expected = [1.29959593, 415.69603238, 1185.50666173]
        expected_values = {
            Dynamic.Vehicle.Propulsion.SHAFT_POWER: (shp_expected, 'hp', 1e-8),
            Dynamic.Vehicle.Propulsion.THRUST: (total_thrust_expected, 'lbf', 1e-8),
            'thrust_summation.propeller_thrust': (prop_thrust_expected, 'lbf', 1e-8),
            Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN: (electric_power_expected, 'kW', 2e-7),
        }

        for var_name, (expected, units, tol) in expected_values.items():
            with self.subTest(var=var_name):
                actual = self.prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tolerance=tol)

        # NOTE: There isn't much point in checking the partials of a component that computes them
        # with FD.
        partial_data = self.prob.check_partials(out_stream=None, form='forward', step=1.01e-6)
        assert_check_partials(partial_data, atol=1e10, rtol=1e-3)

    def test_control_rpm_turboprop(self):
        # based on test_case_2, but simulating RPM as a dymos control
        filename = get_path('models/engines/turboshaft_1120hp.csv')
        test_points = [(0.001, 0, 0), (0, 0, 1), (0.6, 25000, 1)]
        truth_vals = [
            (111.99507922, 37.507376, 910.70245568, 948.20983168, -195.78762),
            (1119.99609612, 136.3, 2752.73508087, 2889.03508087, -644),
            (778.21130479, 21.3, 558.33650216, 579.63650216, -839.7),
        ]

        options = get_option_defaults()
        options.set_val(Aircraft.Engine.DATA_FILE, filename)
        options.delete(Aircraft.Engine.FIXED_RPM)

        self.prepare_model(options, test_points)

        self.prob.set_val(Aircraft.Engine.Propeller.DIAMETER, 10.5, units='ft')
        self.prob.set_val(Aircraft.Engine.Propeller.ACTIVITY_FACTOR, 114.0, units='unitless')
        self.prob.set_val(
            Aircraft.Engine.Propeller.INTEGRATED_LIFT_COEFFICIENT, 0.5, units='unitless'
        )

        self.prob.set_val(Aircraft.Engine.Propeller.TIP_SPEED_MAX, 800, units='ft/s')

        rpm_control = np.array([0.5, 0.75, 1.0]) * 1455.13090827
        self.prob.set_val(
            f'{Dynamic.Vehicle.Propulsion.RPM}_control',
            val=rpm_control,
            units='rpm',
        )

        self.prob.run_model()

        results = self.get_results()

        expected_values = {
            'point_0 (M=0.001, alt=0, idle)': truth_vals[0],
            'point_1 (M=0, alt=0, SLS)': truth_vals[1],
            'point_2 (M=0.6, alt=25k, TOC)': truth_vals[2],
        }

        for point_name, expected in expected_values.items():
            with self.subTest(var=point_name):
                idx = list(expected_values.keys()).index(point_name)
                assert_near_equal(results[idx], expected, tolerance=1.5e-10)

        with self.subTest(var='rotations_per_minute'):
            actual = self.prob.get_val(Dynamic.Vehicle.Propulsion.RPM, units='rpm')
            assert_near_equal(actual, rpm_control, tolerance=1e-8)

        partial_data = self.prob.check_partials(out_stream=None, form='central')
        assert_check_partials(partial_data, atol=0.15, rtol=0.15)


class ExamplePropModel(SubsystemBuilder):
    def build_mission(self, num_nodes, aviary_inputs, user_options, subsystem_options):
        prop_group = om.Group()

        pp = prop_group.add_subsystem(
            'propeller_performance',
            PropellerPerformance(aviary_options=aviary_inputs, num_nodes=num_nodes),
            promotes_inputs=[
                Dynamic.Atmosphere.MACH,
                Aircraft.Engine.Propeller.TIP_SPEED_MAX,
                Aircraft.Engine.Propeller.TIP_MACH_MAX,
                Dynamic.Atmosphere.DENSITY,
                Dynamic.Mission.VELOCITY,
                Aircraft.Engine.Propeller.DIAMETER,
                Aircraft.Engine.Propeller.ACTIVITY_FACTOR,
                Aircraft.Engine.Propeller.INTEGRATED_LIFT_COEFFICIENT,
                Aircraft.Nacelle.AVG_DIAMETER,
                Dynamic.Atmosphere.SPEED_OF_SOUND,
                Dynamic.Vehicle.Propulsion.RPM,
                Dynamic.Vehicle.Propulsion.SHAFT_POWER,
            ],
            promotes_outputs=['*'],
        )

        pp.set_input_defaults(Aircraft.Engine.Propeller.DIAMETER, 10, units='ft')
        pp.set_input_defaults(
            Dynamic.Vehicle.Propulsion.PROPELLER_TIP_SPEED,
            800.0 * np.ones(num_nodes),
            units='ft/s',
        )
        pp.set_input_defaults(Dynamic.Mission.VELOCITY, 100.0 * np.ones(num_nodes), units='knot')

        return prop_group


if __name__ == '__main__':
    unittest.main()
    # test = TurbopropMissionTest()
    # test.setUp()
    # test.test_electroprop_calc_RPM()
