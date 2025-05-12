import unittest
from pathlib import Path

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.subsystems.atmosphere.atmosphere import Atmosphere
from aviary.subsystems.propulsion.motor.motor_builder import MotorBuilder
from aviary.subsystems.propulsion.propeller.propeller_performance import PropellerPerformance
from aviary.subsystems.propulsion.turboprop_model import TurbopropModel
from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
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
        self, test_points=[(0, 0, 0), (0, 0, 1)], shp_model=None, prop_model=None, **kwargs
    ):
        options = get_option_defaults()
        if isinstance(shp_model, Path):
            options.set_val(Aircraft.Engine.DATA_FILE, shp_model)
            shp_model = None
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
        options.set_val(
            Aircraft.Engine.FIXED_RPM,
            1455.13090827,
            units='rpm',
        )

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

        self.prob.model.add_subsystem(
            engine.name,
            subsys=engine.build_mission(num_nodes=num_nodes, aviary_inputs=options, **kwargs),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        setup_model_options(self.prob, options)

        self.prob.setup(force_alloc_complex=False)
        self.prob.set_val(Aircraft.Engine.SCALE_FACTOR, 1, units='unitless')

    def get_results(self, point_names=None, display_results=False):
        shp = self.prob.get_val(Dynamic.Vehicle.Propulsion.SHAFT_POWER, units='hp')
        total_thrust = self.prob.get_val(Dynamic.Vehicle.Propulsion.THRUST, units='lbf')
        prop_thrust = self.prob.get_val('propeller_thrust', units='lbf')
        tailpipe_thrust = self.prob.get_val('turboshaft_thrust', units='lbf')
        max_thrust = self.prob.get_val(Dynamic.Vehicle.Propulsion.THRUST_MAX, units='lbf')
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
                    max_thrust[n],
                    fuel_flow[n],
                )
            )
        return results

    def test_case_1(self):
        # test case using GASP-derived engine deck and "user specified" prop model
        filename = get_path('models/engines/turboshaft_1120hp.deck')
        # Mach, alt, throttle @ idle, SLS, TOC
        test_points = [(0, 0, 0), (0, 0, 1), (0.6, 25000, 1)]
        # shp, tailpipe thrust, prop_thrust, total_thrust, max_thrust, fuel flow
        truth_vals = [
            (
                111.99923788786062,
                37.699999999999996,
                610.3580810058977,
                648.0580810058977,
                4184.157517016291,
                -195.79999999999995,
            ),
            (
                1119.992378878607,
                136.29999999999967,
                4047.857517016292,
                4184.157517016291,
                4184.157517016291,
                -643.9999999999998,
            ),
            (
                778.2106659424866,
                21.30000000000001,
                558.2951237599805,
                579.5951237599804,
                579.5951237599804,
                -839.7000000000685,
            ),
        ]

        options = get_option_defaults()
        options.set_val(
            Aircraft.Engine.Propeller.COMPUTE_INSTALLATION_LOSS,
            val=True,
            units='unitless',
        )
        options.set_val(Aircraft.Engine.Propeller.NUM_BLADES, val=4, units='unitless')
        options.set_val('speed_type', SpeedType.MACH)

        prop_group = ExamplePropModel('custom_prop_model')

        self.prepare_model(test_points, filename, prop_group)

        self.prob.set_val(Aircraft.Engine.Propeller.DIAMETER, 10.5, units='ft')
        self.prob.set_val(Aircraft.Engine.Propeller.ACTIVITY_FACTOR, 114.0, units='unitless')
        # self.prob.set_val(Dynamic.Mission.PERCENT_ROTOR_RPM_CORRECTED,
        #                   np.array([1, 1, 0.7]), units="unitless")
        self.prob.set_val(
            Aircraft.Engine.Propeller.INTEGRATED_LIFT_COEFFICIENT, 0.5, units='unitless'
        )

        self.prob.set_val(Aircraft.Engine.Propeller.TIP_SPEED_MAX, 800, units='ft/s')

        self.prob.run_model()
        results = self.get_results()
        assert_near_equal(results[0], truth_vals[0], tolerance=1.5e-12)
        assert_near_equal(results[1], truth_vals[1], tolerance=1.5e-12)
        assert_near_equal(results[2], truth_vals[2], tolerance=1.5e-12)

        # because Hamilton Standard model uses fd method, the following may not be
        # accurate.
        partial_data = self.prob.check_partials(out_stream=None, form='central')
        assert_check_partials(partial_data, atol=0.2, rtol=0.2)

    def test_case_2(self):
        # test case using GASP-derived engine deck and default HS prop model.
        filename = get_path('models/engines/turboshaft_1120hp.deck')
        test_points = [(0.001, 0, 0), (0, 0, 1), (0.6, 25000, 1)]
        truth_vals = [
            (
                111.99470252,
                37.507375,
                610.74316702,
                648.25054202,
                4174.71017,
                -195.787625,
            ),
            (
                1119.992378878607,
                136.29999999999967,
                4047.857517016292,
                4184.157517016291,
                4184.157517016291,
                -643.9999999999998,
            ),
            (
                778.2106659424866,
                21.30000000000001,
                558.2951237599805,
                579.5951237599804,
                579.5951237599804,
                -839.7000000000685,
            ),
        ]

        self.prepare_model(test_points, filename)

        self.prob.set_val(Aircraft.Engine.Propeller.DIAMETER, 10.5, units='ft')
        self.prob.set_val(Aircraft.Engine.Propeller.ACTIVITY_FACTOR, 114.0, units='unitless')
        # self.prob.set_val(Dynamic.Mission.PERCENT_ROTOR_RPM_CORRECTED,
        #                   np.array([1,1,0.7]), units="unitless")
        self.prob.set_val(
            Aircraft.Engine.Propeller.INTEGRATED_LIFT_COEFFICIENT, 0.5, units='unitless'
        )

        self.prob.set_val(Aircraft.Engine.Propeller.TIP_SPEED_MAX, 800, units='ft/s')

        self.prob.run_model()

        results = self.get_results()
        assert_near_equal(results[0], truth_vals[0], tolerance=1.5e-12)
        assert_near_equal(results[1], truth_vals[1], tolerance=1.5e-12)
        assert_near_equal(results[2], truth_vals[2], tolerance=1.5e-12)

        partial_data = self.prob.check_partials(out_stream=None, form='central')
        assert_check_partials(partial_data, atol=0.15, rtol=0.15)

    def test_case_3(self):
        # test case using GASP-derived engine deck w/o tailpipe thrust and default
        # HS prop model.
        filename = get_path('models/engines/turboshaft_1120hp_no_tailpipe.deck')
        test_points = [(0, 0, 0), (0, 0, 1), (0.6, 25000, 1)]
        truth_vals = [
            (
                111.99923788786062,
                0.0,
                610.3580810058977,
                610.3580810058977,
                4047.857517016292,
                -195.79999999999995,
            ),
            (
                1119.992378878607,
                0.0,
                4047.857517016292,
                4047.857517016292,
                4047.857517016292,
                -643.9999999999998,
            ),
            (
                778.2106659424866,
                0.0,
                558.2951237599805,
                558.2951237599805,
                558.2951237599805,
                -839.7000000000685,
            ),
        ]

        self.prepare_model(test_points, filename)

        self.prob.set_val(Aircraft.Engine.Propeller.DIAMETER, 10.5, units='ft')
        self.prob.set_val(Aircraft.Engine.Propeller.ACTIVITY_FACTOR, 114.0, units='unitless')
        self.prob.set_val(
            Aircraft.Engine.Propeller.INTEGRATED_LIFT_COEFFICIENT, 0.5, units='unitless'
        )
        self.prob.set_val(Aircraft.Engine.Propeller.TIP_SPEED_MAX, 800, units='ft/s')

        self.prob.run_model()

        results = self.get_results()
        assert_near_equal(results[0], truth_vals[0], tolerance=1.5e-12)
        assert_near_equal(results[1], truth_vals[1], tolerance=1.5e-12)
        assert_near_equal(results[2], truth_vals[2], tolerance=1.5e-12)

        # Note: There isn't much point in checking the partials of a component
        # that computes them with FD.
        partial_data = self.prob.check_partials(out_stream=None, form='forward', step=1.01e-6)
        assert_check_partials(partial_data, atol=1e10, rtol=1e-3)

    def test_electroprop(self):
        # test case using electric motor and default HS prop model.
        test_points = [(0, 0, 0), (0, 0, 1), (0.6, 25000, 1)]
        num_nodes = len(test_points)

        motor_model = MotorBuilder()

        self.prepare_model(test_points, motor_model, input_rpm=True)
        self.prob.set_val(Dynamic.Vehicle.Propulsion.RPM, np.ones(num_nodes) * 2000.0, units='rpm')

        self.prob.set_val(Aircraft.Engine.Propeller.DIAMETER, 10.5, units='ft')
        self.prob.set_val(Aircraft.Engine.Propeller.ACTIVITY_FACTOR, 114.0, units='unitless')
        self.prob.set_val(
            Aircraft.Engine.Propeller.INTEGRATED_LIFT_COEFFICIENT, 0.5, units='unitless'
        )

        self.prob.set_val(Aircraft.Engine.Propeller.TIP_SPEED_MAX, 800, units='ft/s')

        self.prob.run_model()

        shp_expected = [0.0, 367.82313837, 367.82313837]
        prop_thrust_expected = total_thrust_expected = [
            610.3580827654595,
            2083.253331913252,
            184.38117745374652,
        ]
        electric_power_expected = [0.0, 303.31014553, 303.31014553]

        shp = self.prob.get_val(Dynamic.Vehicle.Propulsion.SHAFT_POWER, units='hp')
        total_thrust = self.prob.get_val(Dynamic.Vehicle.Propulsion.THRUST, units='lbf')
        prop_thrust = self.prob.get_val('propeller_thrust', units='lbf')
        electric_power = self.prob.get_val(Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN, units='kW')

        assert_near_equal(shp, shp_expected, tolerance=1e-8)
        assert_near_equal(total_thrust, total_thrust_expected, tolerance=1e-8)
        assert_near_equal(prop_thrust, prop_thrust_expected, tolerance=1e-8)
        assert_near_equal(electric_power, electric_power_expected, tolerance=1e-8)

        # Note: There isn't much point in checking the partials of a component
        # that computes them with FD.
        partial_data = self.prob.check_partials(out_stream=None, form='forward', step=1.01e-6)
        assert_check_partials(partial_data, atol=1e10, rtol=1e-3)


class ExamplePropModel(SubsystemBuilderBase):
    def build_mission(self, num_nodes, aviary_inputs, **kwargs):
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
    # test = TurbopropTest()
    # test.setUp()
    # test.test_electroprop()
    # test.test_case_2()
