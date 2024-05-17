import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from dymos.models.atmosphere import USatm1976Comp
from pathlib import Path

from aviary.mission.gasp_based.flight_conditions import FlightConditions
from aviary.subsystems.propulsion.turboprop_model import TurbopropModel
from aviary.subsystems.propulsion.propeller_performance import PropellerPerformance
from aviary.interface.utils.markdown_utils import round_it
from aviary.utils.preprocessors import preprocess_propulsion
from aviary.utils.functions import get_path
from aviary.variable_info.variables import Dynamic, Mission, Settings
from aviary.subsystems.propulsion.motor.motor_variables import Aircraft
from aviary.variable_info.enums import SpeedType, Verbosity
from aviary.variable_info.options import get_option_defaults
from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.subsystems.propulsion.motor.motor_builder import MotorBuilder


class TurbopropTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    def prepare_model(self, test_points=[(0, 0, 0), (0, 0, 1)], shp_model=None, prop_model=None, **kwargs):
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

        options.set_val(Aircraft.Design.COMPUTE_INSTALLATION_LOSS,
                        val=True, units='unitless')
        options.set_val(Aircraft.Engine.NUM_PROPELLER_BLADES,
                        val=4, units='unitless')

        num_nodes = len(test_points)

        engine = TurbopropModel(
            options=options, shaft_power_model=shp_model, propeller_model=prop_model)
        preprocess_propulsion(options, [engine])

        machs, alts, throttles = zip(*test_points)
        IVC = om.IndepVarComp(Dynamic.Mission.MACH,
                              np.array(machs),
                              units='unitless')
        IVC.add_output(Dynamic.Mission.ALTITUDE,
                       np.array(alts),
                       units='ft')
        IVC.add_output(Dynamic.Mission.THROTTLE,
                       np.array(throttles),
                       units='unitless')
        self.prob.model.add_subsystem('IVC', IVC, promotes=['*'])

        self.prob.model.add_subsystem(
            name='atmosphere',
            subsys=USatm1976Comp(num_nodes=num_nodes),
            promotes_inputs=[('h', Dynamic.Mission.ALTITUDE)],
            promotes_outputs=[
                ('sos', Dynamic.Mission.SPEED_OF_SOUND),
                ('rho', Dynamic.Mission.DENSITY),
                ('temp', Dynamic.Mission.TEMPERATURE),
                ('pres', Dynamic.Mission.STATIC_PRESSURE)],)

        # calculate atmospheric properties
        self.prob.model.add_subsystem(
            "flight_conditions",
            FlightConditions(num_nodes=num_nodes, input_speed_type=SpeedType.MACH),
            promotes_inputs=[("rho", Dynamic.Mission.DENSITY),
                             Dynamic.Mission.SPEED_OF_SOUND, 'mach'],
            promotes_outputs=[Dynamic.Mission.DYNAMIC_PRESSURE,
                              'EAS', ('TAS', 'velocity')],
        )

        self.prob.model.add_subsystem(
            engine.name,
            subsys=engine.build_mission(
                num_nodes=num_nodes, aviary_inputs=options, **kwargs),
            promotes_inputs=['*'],
            promotes_outputs=['*'])

        self.prob.setup(force_alloc_complex=True)
        self.prob.set_val(Aircraft.Engine.SCALE_FACTOR, 1, units='unitless')

    def get_results(self, point_names=None, display_results=False):
        shp = self.prob.get_val(Dynamic.Mission.SHAFT_POWER, units='hp')
        total_thrust = self.prob.get_val(Dynamic.Mission.THRUST, units='lbf')
        prop_thrust = self.prob.get_val(
            'turboprop_model.propeller_thrust', units='lbf')
        tailpipe_thrust = self.prob.get_val(
            'turboprop_model.turboshaft_thrust', units='lbf')
        fuel_flow = self.prob.get_val(
            Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE, units='lbm/h')

        results = []
        for n, _ in enumerate(shp):
            results.append(
                (shp[n],
                 tailpipe_thrust[n],
                 prop_thrust[n],
                 total_thrust[n],
                 fuel_flow[n]))
        return results

    def test_case_1(self):
        # test case using GASP-derived engine deck and "user specified" prop model
        filename = get_path('models/engines/turboprop_1120hp.deck')
        # Mach, alt, throttle
        test_points = [(0, 0, 0), (0, 0, 1), (.6, 25000, 1)]
        point_names = ['idle', 'SLS', 'TOC']
        # shp, tailpipe thrust, prop_thrust, total_thrust, fuel flow
        truth_vals = [(112, 37.7, -195.8), (1120, 136.3, -644), (1742.5, 21.3, -839.7)]

        options = get_option_defaults()
        options.set_val(Aircraft.Design.COMPUTE_INSTALLATION_LOSS,
                        val=True, units='unitless')
        options.set_val(Aircraft.Engine.NUM_PROPELLER_BLADES,
                        val=4, units='unitless')
        options.set_val('speed_type', SpeedType.MACH)

        prop_group = ExamplePropModel('custom_prop_model')

        self.prepare_model(test_points, filename, prop_group)

        self.prob.set_val(Aircraft.Engine.PROPELLER_DIAMETER, 10.5, units="ft")
        self.prob.set_val(Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR,
                          114.0, units="unitless")
        self.prob.set_val(
            Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICIENT, 0.5, units="unitless")
        self.prob.set_val(Dynamic.Mission.PERCENT_ROTOR_RPM_CORRECTED,
                          [1.0], units="unitless")
        self.prob.set_val(Aircraft.Design.MAX_PROPELLER_TIP_SPEED,
                          [800.00, 800.0, 750.0], units="ft/s")

        if options.get_val(Settings.VERBOSITY, units='unitless') is Verbosity.DEBUG:
            om.n2(
                self.prob,
                outfile="n2.html",
                show_browser=False,
            )

        self.prob.run_model()
        results = self.get_results(point_names)
        assert_near_equal(results, truth_vals)

    def test_case_2(self):
        # test case using GASP-derived engine deck and default HS prop model.
        filename = get_path('models/engines/turboprop_1120hp.deck')
        test_points = [(0, 0, 0), (0, 0, 1), (.6, 25000, 1)]
        point_names = ['idle', 'SLS', 'TOC']
        truth_vals = [(112, 37.7, -195.8), (1120, 136.3, -644), (1742.5, 21.3, -839.7)]

        self.prepare_model(test_points, filename)

        num_nodes = len(test_points)

        self.prob.set_val(Aircraft.Engine.PROPELLER_DIAMETER, 10.5, units="ft")
        self.prob.set_val(Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR,
                          114.0, units="unitless")
        self.prob.set_val(Dynamic.Mission.PERCENT_ROTOR_RPM_CORRECTED,
                          np.ones(num_nodes), units="unitless")
        self.prob.set_val(
            Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICIENT, 0.5, units="unitless")
        self.prob.set_val(Dynamic.Mission.PROPELLER_TIP_SPEED,
                          800.*np.ones(num_nodes), units="ft/s")

        self.prob.set_val(Aircraft.Design.MAX_PROPELLER_TIP_SPEED,
                          [800.00, 800.0, 750.0], units="ft/s")

        self.prob.run_model()
        results = self.get_results(point_names)
        assert_near_equal(results, truth_vals)

    def test_case_3(self):
        # test case using GASP-derived engine deck w/o tailpipe thrust and default HS prop model.
        filename = get_path('models/engines/turboprop_1120hp_no_tailpipe.deck')
        test_points = [(0, 0, 0), (0, 0, 1), (.6, 25000, 1)]
        point_names = ['idle', 'SLS', 'TOC']
        truth_vals = [(112, 0, -195.8), (1120, 0, -644), (1742.5, 0, -839.7)]

        self.prepare_model(filename, test_points)

        num_nodes = len(test_points)

        self.prob.set_val(Aircraft.Engine.PROPELLER_DIAMETER, 10.5, units="ft")
        self.prob.set_val(Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR,
                          114.0, units="unitless")
        self.prob.set_val(
            Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICIENT, 0.5, units="unitless")
        self.prob.set_val(Dynamic.Mission.PROPELLER_TIP_SPEED,
                          800.*np.ones(num_nodes), units="ft/s")

        self.prob.run_model()
        results = self.get_results(point_names)
        assert_near_equal(results, truth_vals)

    def test_electroprop(self):
        # test case using GASP-derived engine deck and default HS prop model.
        test_points = [(0, 0, 0), (0, 0, 1), (.6, 25000, 1)]
        num_nodes = len(test_points)

        motor_model = MotorBuilder()

        self.prepare_model(test_points, motor_model, input_rpm=True)
        self.prob.set_val(Aircraft.Motor.RPM, np.ones(num_nodes)*2000., units='rpm')

        self.prob.set_val(Aircraft.Engine.PROPELLER_DIAMETER, 10.5, units="ft")
        self.prob.set_val(Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR,
                          114.0, units="unitless")
        self.prob.set_val(
            Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICIENT, 0.5, units="unitless")
        self.prob.set_val(Dynamic.Mission.PROPELLER_TIP_SPEED,
                          800.*np.ones(num_nodes), units="ft/s")

        self.prob.run_model()

        shp_expected = [0., 505.55333, 505.55333]
        tailpipe_thrust_expected = [0, 0, 0]
        prop_thrust_expected = total_thrust_expected = [1, 1, 1]
        fuel_flow_expected = [0, 0, 0]
        electric_power_expected = [0.0, 408.4409047, 408.4409047]

        shp = self.prob.get_val(Dynamic.Mission.SHAFT_POWER, units='hp')
        total_thrust = self.prob.get_val(Dynamic.Mission.THRUST, units='lbf')
        prop_thrust = self.prob.get_val(
            'turboprop_model.propeller_thrust', units='lbf')
        tailpipe_thrust = self.prob.get_val(
            'turboprop_model.turboshaft_thrust', units='lbf')
        fuel_flow = self.prob.get_val(
            Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE, units='lbm/h')
        electric_power = self.prob.get_val(Dynamic.Mission.ELECTRIC_POWER, units='kW')

        assert_near_equal(shp, shp_expected, tolerance=1e-8)
        assert_near_equal(total_thrust, total_thrust_expected, tolerance=1e-8)
        assert_near_equal(prop_thrust, prop_thrust_expected, tolerance=1e-8)
        assert_near_equal(tailpipe_thrust, tailpipe_thrust_expected, tolerance=1e-8)
        assert_near_equal(fuel_flow, fuel_flow_expected, tolerance=1e-8)
        assert_near_equal(electric_power, electric_power_expected, tolerance=1e-8)


class ExamplePropModel(SubsystemBuilderBase):
    def build_mission(self, num_nodes, aviary_inputs, **kwargs):
        prop_group = om.Group()

        pp = prop_group.add_subsystem(
            'propeller_performance',
            PropellerPerformance(aviary_options=aviary_inputs, num_nodes=num_nodes),
            promotes_inputs=[Dynamic.Mission.TEMPERATURE,
                             Dynamic.Mission.MACH,
                             Dynamic.Mission.PERCENT_ROTOR_RPM_CORRECTED,
                             Aircraft.Design.MAX_PROPELLER_TIP_SPEED,
                             Dynamic.Mission.DENSITY,
                             Dynamic.Mission.VELOCITY,
                             Aircraft.Engine.PROPELLER_DIAMETER,
                             Dynamic.Mission.SHAFT_POWER,
                             Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR,
                             Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICIENT
                             ],
            promotes_outputs=['*'],
        )

        pp.set_input_defaults(Aircraft.Engine.PROPELLER_DIAMETER, 10, units="ft")
        pp.set_input_defaults(Dynamic.Mission.PROPELLER_TIP_SPEED,
                              800.*np.ones(num_nodes), units="ft/s")
        pp.set_input_defaults(Dynamic.Mission.VELOCITY, 100. *
                              np.ones(num_nodes), units="knot")
        pp.set_input_defaults(Dynamic.Mission.TEMPERATURE, 500. *
                              np.ones(num_nodes), units="degR")

        return prop_group


if __name__ == "__main__":
    # unittest.main()
    test = TurbopropTest()
    test.setUp()
    # test.test_electroprop()
    test.test_case_2()
