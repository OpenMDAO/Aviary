import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from dymos.models.atmosphere import USatm1976Comp

from aviary.mission.gasp_based.flight_conditions import FlightConditions
from aviary.subsystems.propulsion.turboprop_model import TurbopropModel
from aviary.subsystems.propulsion.propeller_performance import PropellerPerformance
from aviary.interface.utils.markdown_utils import round_it
from aviary.utils.preprocessors import preprocess_propulsion
from aviary.utils.functions import get_path
from aviary.variable_info.variables import Settings
from aviary.subsystems.propulsion.motor.motor_variables import Aircraft, Dynamic
from aviary.variable_info.enums import SpeedType, Verbosity
from aviary.variable_info.options import get_option_defaults
from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.subsystems.propulsion.motor.motor_builder import MotorBuilder


class ElectropropTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    def prepare_model(self, test_points=[(0, 0, 0), (0, 0, 1)], prop_model=None):
        options = get_option_defaults()
        # options.set_val(Aircraft.Motor.COUNT, 1)

        num_nodes = len(test_points)

        engine = TurbopropModel(
            options=options, shaft_power_model=MotorBuilder(), propeller_model=prop_model)
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
                ('sos', Dynamic.Mission.SPEED_OF_SOUND), ('rho', Dynamic.Mission.DENSITY),
                ('temp', Dynamic.Mission.TEMPERATURE), ('pres', Dynamic.Mission.STATIC_PRESSURE)],
        )

        self.prob.model.add_subsystem(
            engine.name,
            subsys=engine.build_mission(
                num_nodes=num_nodes, aviary_inputs=options),
            promotes_inputs=['*'],
            promotes_outputs=['*'])

        self.prob.setup(force_alloc_complex=True)
        self.prob.set_val(Aircraft.Engine.SCALE_FACTOR, 1, units='unitless')
        self.prob.set_val(Aircraft.Motor.RPM, 2000., units='rpm')

    def test_case_1(self):
        # test case using GASP-derived engine deck and "user specified" prop model
        test_points = [(0, 0, 0), (0, 0, 1), (.6, 25000, 1)]
        point_names = ['idle', 'SLS', 'TOC']

        options = get_option_defaults()
        options.set_val(Aircraft.Design.COMPUTE_INSTALLATION_LOSS,
                        val=True, units='unitless')
        options.set_val(Aircraft.Engine.NUM_PROPELLER_BLADES,
                        val=4, units='unitless')
        options.set_val('speed_type', SpeedType.MACH)

        self.prepare_model(test_points)

        self.prob.set_val(Aircraft.Engine.PROPELLER_DIAMETER, 10.5, units="ft")
        self.prob.set_val(Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR,
                          114.0, units="unitless")
        self.prob.set_val(
            Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICIENT, 0.5, units="unitless")

        om.n2(
            self.prob,
            outfile="n2.html",
            show_browser=False,
        )

        self.prob.run_model()

        self.prob.model.list_inputs(print_arrays=True, units=True)
        self.prob.model.list_outputs(print_arrays=True, units=True)


if __name__ == "__main__":
    unittest.main()
