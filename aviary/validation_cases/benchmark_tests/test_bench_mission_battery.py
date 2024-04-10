import unittest

import numpy as np
import openmdao.api as om

from numpy.testing import assert_almost_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.variable_info.enums import Verbosity
from aviary.variable_info.variables import Aircraft, Dynamic
from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.subsystems.energy.battery_builder import BatteryBuilder


@use_tempdirs
class TestSubsystemsMission(unittest.TestCase):
    def setUp(self):
        self.phase_info = {
            'cruise': {
                "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
                'external_subsystems': [DummyPowerComp('power_generator'), BatteryBuilder('battery')],
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
            "models/test_aircraft/aircraft_for_bench_GwFm.csv", self.phase_info)

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

        prob.phase_info['cruise']['initial_guesses'][f'states:{Mission.Dummy.VARIABLE}'] = ([
            10., 100.], 'm')
        prob.set_initial_guesses()

        # add an assert to see if the initial guesses are correct for Mission.Dummy.VARIABLE
        assert_almost_equal(prob[f'traj.phases.cruise.states:{Mission.Dummy.VARIABLE}'], [[10.],
                                                                                          [25.97729616],
                                                                                          [48.02270384],
                                                                                          [55.],
                                                                                          [70.97729616],
                                                                                          [93.02270384],
                                                                                          [100.]])

        prob.run_aviary_problem()

        # add an assert to see if MoreMission.Dummy.TIMESERIES_VAR was correctly added to the dymos problem
        assert_almost_equal(prob[f'traj.phases.cruise.timeseries.{MoreMission.Dummy.TIMESERIES_VAR}'], np.array(
            [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]).T)


class DummyPowerComp(SubsystemBuilderBase):
    def build_mission(self, num_nodes=1, aviary_inputs=None):
        mission_group = om.Group()
        comp = om.IndepVarComp('power', val=np.ones(num_nodes)*1000, units='kW')
        mission_group.add_subsystem('dummy_comp', subsys=comp, promotes=[
                                    ('power', Dynamic.Mission.ELECTRIC_POWER_TOTAL)])

        return mission_group


if __name__ == "__main__":
    # unittest.main()
    test = TestSubsystemsMission()
    test.setUp()
    test.test_HE_mission()
