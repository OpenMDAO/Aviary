import unittest

import numpy as np
import openmdao.api as om

from numpy.testing import assert_almost_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.variable_info.enums import Verbosity
from aviary.variable_info.variables import Aircraft, Dynamic
from aviary.subsystems.propulsion.engine_model import EngineModel
from aviary.subsystems.energy.battery_builder import BatteryBuilder
from aviary.utils.preprocessors import preprocess_propulsion


@use_tempdirs
class TestSubsystemsMission(unittest.TestCase):
    def setUp(self):
        self.phase_info = {
            'cruise': {
                "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
                'external_subsystems': [BatteryBuilder('battery')],
                "user_options": {
                    "optimize_mach": False,
                    "optimize_altitude": False,
                    "polynomial_control_order": 1,
                    "num_segments": 2,
                    "order": 3,
                    "solve_for_distance": False,
                    "initial_mach": (0.72, "unitless"),
                    "final_mach": (0.72, "unitless"),
                    "mach_bounds": ((0.7, 0.74), "unitless"),
                    "initial_altitude": (35000.0, "ft"),
                    "final_altitude": (35000.0, "ft"),
                    "altitude_bounds": ((23000.0, 38000.0), "ft"),
                    "throttle_enforcement": "boundary_constraint",
                    "fix_initial": True,
                    "constrain_final": False,
                    "fix_duration": False,
                    "initial_bounds": ((0.0, 0.0), "min"),
                    "duration_bounds": ((10., 30.), "min"),
                },
                "initial_guesses": {"times": ([0, 30], "min")},
            }}

    def test_HE_mission(self):
        prob = AviaryProblem()

        prob.load_inputs(
            "models/test_aircraft/aircraft_for_bench_GwFm.csv",
            self.phase_info)

        aviary_inputs = prob.aviary_inputs
        engine_models = aviary_inputs.get_val('engine_models')
        engine_models.append(DummyPowerComp(options=aviary_inputs))
        aviary_inputs.set_val('engine_models', engine_models)
        preprocess_propulsion(aviary_inputs, engine_models)

        prob.aviary_inputs.set_val(Aircraft.Battery.DISCHARGE_LIMIT, 0.2)

        # Preprocess inputs
        prob.check_and_preprocess_inputs()

        prob.add_pre_mission_systems()

        prob.add_phases()

        prob.add_post_mission_systems()

        # Link phases and variables
        prob.link_phases()

        prob.add_driver("SLSQP", max_iter=0, verbosity=Verbosity.QUIET)

        prob.add_design_variables()

        prob.add_objective('fuel_burned')

        prob.setup()

        prob.set_initial_guesses()

        prob.run_aviary_problem()


class DummyPowerComp(EngineModel):
    # A dummy component that adds electric power demand to the otherwise all fuel-burning aircraft
    def build_mission(self, num_nodes=1, aviary_inputs=None):
        comp = om.ExplicitComponent()
        comp.add_input(Dynamic.Mission.ALTITUDE, val=np.zeros(num_nodes), units='ft')
        comp.add_input(Dynamic.Mission.MACH, val=np.zeros(num_nodes), units='unitless')
        comp.add_input(Dynamic.Mission.THROTTLE,
                       val=np.zeros(num_nodes), units='unitless')

        comp.add_output(Dynamic.Mission.ELECTRIC_POWER,
                        val=np.ones(num_nodes)*10, units='kW')
        comp.add_output(Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE,
                        val=np.zeros(num_nodes), units='lbm/h')
        comp.add_output(Dynamic.Mission.NOX_RATE, val=np.zeros(num_nodes), units='lbm/h')
        comp.add_output(Dynamic.Mission.THRUST, val=np.zeros(num_nodes), units='lbf')
        comp.add_output(Dynamic.Mission.THRUST_MAX, val=np.zeros(num_nodes), units='lbf')

        return comp


if __name__ == "__main__":
    # unittest.main()
    test = TestSubsystemsMission()
    test.setUp()
    test.test_HE_mission()
