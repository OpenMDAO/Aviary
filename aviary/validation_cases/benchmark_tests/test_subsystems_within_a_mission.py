import unittest

import numpy as np
from numpy.testing import assert_almost_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.subsystems.test.test_dummy_subsystem import (
    AdditionalArrayGuessSubsystemBuilder,
    ArrayGuessSubsystemBuilder,
    Mission,
    MoreMission,
    PostOnlyBuilder,
)


@use_tempdirs
class TestSubsystemsMission(unittest.TestCase):
    """Test the setup and run of a model with external subsystem."""

    def setUp(self):
        self.phase_info = {
            'pre_mission': {
                'include_takeoff': False,
                'external_subsystems': [
                    ArrayGuessSubsystemBuilder(),
                    AdditionalArrayGuessSubsystemBuilder(),
                ],
                'optimize_mass': True,
            },
            'cruise': {
                'subsystem_options': {
                    'core_aerodynamics': {'method': 'cruise', 'solve_alpha': True}
                },
                'external_subsystems': [
                    ArrayGuessSubsystemBuilder(),
                    AdditionalArrayGuessSubsystemBuilder(),
                ],
                'user_options': {
                    'num_segments': 2,
                    'order': 3,
                    'mach_optimize': False,
                    'mach_polynomial_order': 1,
                    'mach_initial': (0.72, 'unitless'),
                    'mach_final': (0.72, 'unitless'),
                    'mach_bounds': ((0.7, 0.74), 'unitless'),
                    'altitude_optimize': False,
                    'altitude_polynomial_order': 1,
                    'altitude_initial': (35000.0, 'ft'),
                    'altitude_final': (35000.0, 'ft'),
                    'altitude_bounds': ((23000.0, 38000.0), 'ft'),
                    'throttle_enforcement': 'boundary_constraint',
                    'time_initial': (0.0, 'min'),
                    'time_duration_bounds': ((10.0, 30.0), 'min'),
                },
                'initial_guesses': {'time': ([0, 30], 'min')},
            },
            'post_mission': {
                'include_landing': False,
                'external_subsystems': [PostOnlyBuilder()],
            },
        }

    def test_subsystems_in_a_mission(self):
        phase_info = self.phase_info.copy()

        prob = AviaryProblem(verbosity=0)

        prob.load_inputs('models/aircraft/test_aircraft/aircraft_for_bench_GwFm.csv', phase_info)

        prob.check_and_preprocess_inputs()

        prob.build_model()

        prob.model.mission_info['cruise']['initial_guesses'][f'states:{Mission.Dummy.VARIABLE}'] = (
            [10.0, 100.0],
            'm',
        )

        prob.add_driver('SLSQP', max_iter=0, verbosity=0)

        prob.add_design_variables()

        prob.add_objective('fuel_burned')

        prob.setup()

        # add an assert to see if the initial guesses are correct for Mission.Dummy.VARIABLE
        assert_almost_equal(
            prob.get_val(f'traj.cruise.states:{Mission.Dummy.VARIABLE}'),
            [[10.0], [25.97729616], [48.02270384], [55.0], [70.97729616], [93.02270384], [100.0]],
        )

        prob.run_aviary_problem()

        # add an assert to see if MoreMission.Dummy.TIMESERIES_VAR was correctly added to the dymos problem
        assert_almost_equal(
            prob[f'traj.phases.cruise.timeseries.{MoreMission.Dummy.TIMESERIES_VAR}'],
            np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]).T,
        )

    def test_bad_initial_guess_key(self):
        phase_info = self.phase_info.copy()
        phase_info['cruise']['initial_guesses']['bad_guess_name'] = ([10.0, 100.0], 'm')

        prob = AviaryProblem(reports=False, verbosity=0)

        prob.load_inputs('models/aircraft/test_aircraft/aircraft_for_bench_GwFm.csv', phase_info)

        prob.check_and_preprocess_inputs()

        prob.add_pre_mission_systems()

        with self.assertRaises(TypeError) as context:
            prob.add_phases()
        # print(str(context.exception))
        self.assertTrue(
            'EnergyPhase: cruise: unsupported initial guess: bad_guess_name'
            in str(context.exception)
        )


if __name__ == '__main__':
    unittest.main()
