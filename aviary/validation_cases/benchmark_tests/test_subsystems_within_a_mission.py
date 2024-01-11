import numpy as np
import unittest

import pkg_resources
from numpy.testing import assert_almost_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.interface.methods_for_level2 import AviaryProblem

from aviary.subsystems.test.test_dummy_subsystem import (
    PostOnlyBuilder, ArrayGuessSubsystemBuilder, AdditionalArrayGuessSubsystemBuilder,
    Mission, MoreMission)


@use_tempdirs
class TestSubsystemsMission(unittest.TestCase):
    def setUp(self):
        polar_file = "subsystems/aerodynamics/gasp_based/data/large_single_aisle_1_aero_free.txt"
        aero_data = pkg_resources.resource_filename("aviary", polar_file)

        self.phase_info = {
            'pre_mission': {
                'include_takeoff': False,
                'external_subsystems': [ArrayGuessSubsystemBuilder(), AdditionalArrayGuessSubsystemBuilder()],
                'optimize_mass': False,
            },
            'cruise': {
                'subsystem_options': {
                    'core_aerodynamics': {'method': 'solved_alpha', 'aero_data': aero_data, 'training_data': False}
                },
                'external_subsystems': [ArrayGuessSubsystemBuilder(), AdditionalArrayGuessSubsystemBuilder()],
                'user_options': {
                    'fix_initial': True,
                    'fix_final': False,
                    'fix_duration': False,
                    'num_segments': 1,
                    'order': 5,
                    'initial_ref': (1., 's'),
                    'initial_bounds': ((0., 0.), 's'),
                    'duration_ref': (21.e3, 's'),
                    'duration_bounds': ((10.e3, 20.e3), 's'),
                    'min_altitude': (10.668e3, 'm'),
                    'max_altitude': (10.668e3, 'm'),
                    'min_mach': 0.8,
                    'max_mach': 0.8,
                    'required_available_climb_rate': (1.524, 'm/s'),
                    'input_initial': False,
                    'mass_f_cruise': (1.e4, 'lbm'),
                    'range_f_cruise': (1.e6, 'm'),
                },
                'initial_guesses': {
                    'times': ([0., 15000.], 's'),
                    'altitude': ([35.e3, 35.e3], 'ft'),
                    'velocity': ([455.49, 455.49], 'kn'),
                    'mass': ([130.e3, 120.e3], 'lbm'),
                    'range': ([0., 3000.], 'NM'),
                    'velocity_rate': ([0., 0.], 'm/s**2'),
                    'throttle': ([0.6, 0.6], 'unitless'),
                }
            },
            'post_mission': {
                'include_landing': False,
                'external_subsystems': [PostOnlyBuilder()],
            }
        }

    def test_subsystems_in_a_mission(self):
        phase_info = self.phase_info.copy()

        prob = AviaryProblem()

        csv_path = pkg_resources.resource_filename(
            "aviary", "models/test_aircraft/aircraft_for_bench_GwFm.csv")

        prob.load_inputs(csv_path, phase_info)

        # Have checks for clashing user inputs
        # Raise warnings or errors depending on how clashing the issues are
        prob.check_inputs()

        prob.add_pre_mission_systems()

        prob.add_phases()

        prob.add_post_mission_systems()

        # Link phases and variables
        prob.link_phases()

        prob.add_driver("SLSQP", max_iter=0)

        prob.add_design_variables()

        prob.add_objective('fuel')

        prob.setup()

        prob.phase_info['cruise']['initial_guesses'][f'states:{Mission.Dummy.VARIABLE}'] = ([
            10., 100.], 'm')
        prob.set_initial_guesses()

        # add an assert to see if the initial guesses are correct for Mission.Dummy.VARIABLE
        assert_almost_equal(prob[f'traj.phases.cruise.states:{Mission.Dummy.VARIABLE}'], [[10.],
                                                                                          [22.57838779],
                                                                                          [47.47686109],
                                                                                          [75.08412877],
                                                                                          [94.86062235],
                                                                                          [100.],])

        prob.run_aviary_problem()

        # add an assert to see if MoreMission.Dummy.TIMESERIES_VAR was correctly added to the dymos problem
        assert_almost_equal(prob[f'traj.phases.cruise.timeseries.{MoreMission.Dummy.TIMESERIES_VAR}'], np.array(
            [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]).T)

    def test_bad_initial_guess_key(self):
        phase_info = self.phase_info.copy()
        phase_info['cruise']['initial_guesses']['bad_guess_name'] = ([10., 100.], 'm')

        prob = AviaryProblem(reports=False)

        csv_path = pkg_resources.resource_filename(
            "aviary", "models/test_aircraft/aircraft_for_bench_GwFm.csv")

        prob.load_inputs(csv_path, phase_info)

        # Have checks for clashing user inputs
        # Raise warnings or errors depending on how clashing the issues are
        prob.check_inputs()

        prob.add_pre_mission_systems()

        with self.assertRaises(TypeError) as context:
            prob.add_phases()
        self.assertTrue(
            'Cruise: cruise: unsupported initial guess: bad_guess_name' in str(context.exception))


if __name__ == "__main__":
    # unittest.main()
    test = TestSubsystemsMission()
    test.setUp()
    test.test_subsystems_in_a_mission()
