import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from aviary.subsystems.atmosphere.atmosphere import Atmosphere
from pathlib import Path
from openmdao.utils.testing_utils import use_tempdirs

from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.subsystems.propulsion.turboprop_model import TurbopropModel
from aviary.subsystems.propulsion.propeller.propeller_performance import (
    PropellerPerformance,
)
from aviary.utils.preprocessors import preprocess_propulsion
from aviary.utils.functions import get_path
from aviary.variable_info.variables import Aircraft, Dynamic, Mission
from aviary.variable_info.enums import SpeedType
from aviary.variable_info.options import get_option_defaults
from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.subsystems.propulsion.motor.motor_builder import MotorBuilder


@use_tempdirs
class TurbopropTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    def prepare_model(
        self,
        test_points=[(0, 0, 0), (0, 0, 1)],
        shp_model=None,
        prop_model=None,
        **kwargs
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
            Aircraft.Engine.COMPUTE_PROPELLER_INSTALLATION_LOSS,
            val=True,
            units='unitless',
        )
        options.set_val(Aircraft.Engine.NUM_PROPELLER_BLADES, val=4, units='unitless')

        num_nodes = len(test_points)

        engine = TurbopropModel(
            options=options, shaft_power_model=shp_model, propeller_model=prop_model
        )
        preprocess_propulsion(options, [engine])

        machs, alts, throttles = zip(*test_points)
        IVC = om.IndepVarComp(Dynamic.Mission.MACH, np.array(machs), units='unitless')
        IVC.add_output(Dynamic.Mission.ALTITUDE, np.array(alts), units='ft')
        IVC.add_output(Dynamic.Mission.THROTTLE, np.array(throttles), units='unitless')
        self.prob.model.add_subsystem('IVC', IVC, promotes=['*'])

        # calculate atmospheric properties
        self.prob.model.add_subsystem(
            name='atmosphere',
            subsys=Atmosphere(num_nodes=num_nodes, input_speed_type=SpeedType.MACH),
            promotes=['*'],
        )

        self.prob.model.add_subsystem(
            engine.name,
            subsys=engine.build_mission(
                num_nodes=num_nodes, aviary_inputs=options, **kwargs
            ),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        self.prob.setup(force_alloc_complex=False)
        self.prob.set_val(Aircraft.Engine.SCALE_FACTOR, 1, units='unitless')

    def get_results(self, point_names=None, display_results=False):
        shp = self.prob.get_val(Dynamic.Mission.SHAFT_POWER, units='hp')
        total_thrust = self.prob.get_val(Dynamic.Mission.THRUST, units='lbf')
        prop_thrust = self.prob.get_val('turboprop_model.propeller_thrust', units='lbf')
        tailpipe_thrust = self.prob.get_val(
            'turboprop_model.turboshaft_thrust', units='lbf'
        )
        max_thrust = self.prob.get_val(Dynamic.Mission.THRUST_MAX, units='lbf')
        fuel_flow = self.prob.get_val(
            Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE, units='lbm/h'
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
                223.99923788786057,
                37.699999999999996,
                1195.4410222483584,
                1233.1410222483585,
                4983.816420783667,
                -195.79999999999995,
            ),
            (
                2239.9923788786077,
                136.29999999999967,
                4847.516420783668,
                4983.816420783667,
                4983.816420783667,
                -643.9999999999998,
            ),
            (
                2466.55094358958,
                21.30000000000001,
                1834.6578916888234,
                1855.9578916888233,
                1855.9578916888233,
                -839.7000000000685,
            ),
        ]

        options = get_option_defaults()
        options.set_val(
            Aircraft.Engine.COMPUTE_PROPELLER_INSTALLATION_LOSS,
            val=True,
            units='unitless',
        )
        options.set_val(Aircraft.Engine.NUM_PROPELLER_BLADES, val=4, units='unitless')
        options.set_val('speed_type', SpeedType.MACH)

        prop_group = ExamplePropModel('custom_prop_model')

        self.prepare_model(test_points, filename, prop_group)

        self.prob.set_val(Aircraft.Engine.PROPELLER_DIAMETER, 10.5, units="ft")
        self.prob.set_val(
            Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR, 114.0, units="unitless"
        )
        # self.prob.set_val(Dynamic.Mission.PERCENT_ROTOR_RPM_CORRECTED,
        #                   np.array([1, 1, 0.7]), units="unitless")
        self.prob.set_val(
            Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICIENT, 0.5, units="unitless"
        )

        self.prob.set_val(Aircraft.Engine.PROPELLER_TIP_SPEED_MAX, 800, units="ft/s")

        self.prob.run_model()
        results = self.get_results()
        assert_near_equal(results[0], truth_vals[0], tolerance=1.5e-12)
        assert_near_equal(results[1], truth_vals[1], tolerance=1.5e-12)
        assert_near_equal(results[2], truth_vals[2], tolerance=1.5e-12)

        # because Hamilton Standard model uses fd method, the following may not be accurate.
        partial_data = self.prob.check_partials(out_stream=None, form="central")
        assert_check_partials(partial_data, atol=0.2, rtol=0.2)

    def test_case_2(self):
        # test case using GASP-derived engine deck and default HS prop model.
        filename = get_path('models/engines/turboshaft_1120hp.deck')
        test_points = [(0.001, 0, 0), (0, 0, 1), (0.6, 25000, 1)]
        truth_vals = [
            (
                223.99007751511726,
                37.507374999999996,
                1186.7060713100836,
                1224.2134463100836,
                4984.168016782585,
                -195.78762499999996,
            ),
            (
                2239.9923788786077,
                136.29999999999967,
                4847.516420783668,
                4983.816420783667,
                4983.816420783667,
                -643.9999999999998,
            ),
            (
                2466.55094358958,
                21.30000000000001,
                1834.6578916888234,
                1855.9578916888233,
                1855.9578916888233,
                -839.7000000000685,
            ),
        ]

        self.prepare_model(test_points, filename)

        self.prob.set_val(Aircraft.Engine.PROPELLER_DIAMETER, 10.5, units="ft")
        self.prob.set_val(
            Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR, 114.0, units="unitless"
        )
        # self.prob.set_val(Dynamic.Mission.PERCENT_ROTOR_RPM_CORRECTED,
        #                   np.array([1,1,0.7]), units="unitless")
        self.prob.set_val(
            Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICIENT, 0.5, units="unitless"
        )

        self.prob.set_val(Aircraft.Engine.PROPELLER_TIP_SPEED_MAX, 800, units="ft/s")

        self.prob.run_model()

        results = self.get_results()
        assert_near_equal(results[0], truth_vals[0], tolerance=1.5e-12)
        assert_near_equal(results[1], truth_vals[1], tolerance=1.5e-12)
        assert_near_equal(results[2], truth_vals[2], tolerance=1.5e-12)

        partial_data = self.prob.check_partials(out_stream=None, form="central")
        assert_check_partials(partial_data, atol=0.15, rtol=0.15)

    def test_case_3(self):
        # test case using GASP-derived engine deck w/o tailpipe thrust and default HS prop model.
        filename = get_path('models/engines/turboshaft_1120hp_no_tailpipe.deck')
        test_points = [(0, 0, 0), (0, 0, 1), (0.6, 25000, 1)]
        truth_vals = [
            (
                223.99923788786057,
                0.0,
                1195.4410222483584,
                1195.4410222483584,
                4847.516420783668,
                -195.79999999999995,
            ),
            (
                2239.9923788786077,
                0.0,
                4847.516420783668,
                4847.516420783668,
                4847.516420783668,
                -643.9999999999998,
            ),
            (
                2466.55094358958,
                0.0,
                1834.6578916888234,
                1834.6578916888234,
                1834.6578916888234,
                -839.7000000000685,
            ),
        ]

        self.prepare_model(test_points, filename)

        self.prob.set_val(Aircraft.Engine.PROPELLER_DIAMETER, 10.5, units="ft")
        self.prob.set_val(
            Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR, 114.0, units="unitless"
        )
        self.prob.set_val(
            Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICIENT, 0.5, units="unitless"
        )
        self.prob.set_val(Aircraft.Engine.PROPELLER_TIP_SPEED_MAX, 800, units="ft/s")

        self.prob.run_model()

        results = self.get_results()
        assert_near_equal(results[0], truth_vals[0], tolerance=1.5e-12)
        assert_near_equal(results[1], truth_vals[1], tolerance=1.5e-12)
        assert_near_equal(results[2], truth_vals[2], tolerance=1.5e-12)

        partial_data = self.prob.check_partials(out_stream=None, form="central")
        assert_check_partials(partial_data, atol=0.15, rtol=0.15)

    def test_electroprop(self):
        # test case using electric motor and default HS prop model.
        test_points = [(0, 0, 0), (0, 0, 1), (0.6, 25000, 1)]
        num_nodes = len(test_points)

        motor_model = MotorBuilder()

        self.prepare_model(test_points, motor_model, input_rpm=True)
        self.prob.set_val(Dynamic.Mission.RPM, np.ones(num_nodes) * 2000.0, units='rpm')

        self.prob.set_val(Aircraft.Engine.PROPELLER_DIAMETER, 10.5, units="ft")
        self.prob.set_val(
            Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR, 114.0, units="unitless"
        )
        self.prob.set_val(
            Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICIENT, 0.5, units="unitless"
        )

        self.prob.set_val(Aircraft.Engine.PROPELLER_TIP_SPEED_MAX, 800, units="ft/s")

        self.prob.run_model()

        shp_expected = [0.0, 505.55333, 505.55333]
        prop_thrust_expected = total_thrust_expected = [
            610.35808276,
            2627.2632965,
            312.64111293,
        ]
        electric_power_expected = [0.0, 408.4409047, 408.4409047]

        shp = self.prob.get_val(Dynamic.Mission.SHAFT_POWER, units='hp')
        total_thrust = self.prob.get_val(Dynamic.Mission.THRUST, units='lbf')
        prop_thrust = self.prob.get_val('turboprop_model.propeller_thrust', units='lbf')
        electric_power = self.prob.get_val(
            Dynamic.Mission.ELECTRIC_POWER_IN, units='kW'
        )

        assert_near_equal(shp, shp_expected, tolerance=1e-8)
        assert_near_equal(total_thrust, total_thrust_expected, tolerance=1e-8)
        assert_near_equal(prop_thrust, prop_thrust_expected, tolerance=1e-8)
        assert_near_equal(electric_power, electric_power_expected, tolerance=1e-8)

        partial_data = self.prob.check_partials(
            out_stream=None, method="fd", form="central"
        )
        assert_check_partials(partial_data, atol=0.17, rtol=0.15)


class ExamplePropModel(SubsystemBuilderBase):
    def build_mission(self, num_nodes, aviary_inputs, **kwargs):
        prop_group = om.Group()

        pp = prop_group.add_subsystem(
            'propeller_performance',
            PropellerPerformance(aviary_options=aviary_inputs, num_nodes=num_nodes),
            promotes_inputs=[
                Dynamic.Mission.MACH,
                Aircraft.Engine.PROPELLER_TIP_SPEED_MAX,
                Aircraft.Engine.PROPELLER_TIP_MACH_MAX,
                Dynamic.Mission.DENSITY,
                Dynamic.Mission.VELOCITY,
                Aircraft.Engine.PROPELLER_DIAMETER,
                Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR,
                Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICIENT,
                Aircraft.Nacelle.AVG_DIAMETER,
                Dynamic.Mission.SPEED_OF_SOUND,
                Dynamic.Mission.RPM,
                Dynamic.Mission.SHAFT_POWER,
            ],
            promotes_outputs=['*'],
        )

        pp.set_input_defaults(Aircraft.Engine.PROPELLER_DIAMETER, 10, units="ft")
        pp.set_input_defaults(
            Dynamic.Mission.PROPELLER_TIP_SPEED,
            800.0 * np.ones(num_nodes),
            units="ft/s",
        )
        pp.set_input_defaults(
            Dynamic.Mission.VELOCITY, 100.0 * np.ones(num_nodes), units="knot"
        )

        return prop_group


if __name__ == "__main__":
    unittest.main()
    # test = TurbopropTest()
    # test.setUp()
    # test.test_electroprop()
    # test.test_case_2()
