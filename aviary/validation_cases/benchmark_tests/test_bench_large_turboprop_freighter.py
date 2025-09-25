import unittest
from copy import deepcopy

from numpy.testing import assert_almost_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.models.aircraft.large_turboprop_freighter.phase_info import (
    energy_phase_info,
    two_dof_phase_info,
)
from aviary.subsystems.propulsion.turboprop_model import TurbopropModel
from aviary.utils.process_input_decks import create_vehicle
from aviary.variable_info.variables import Aircraft, Mission, Settings


@use_tempdirs
# TODO need to add asserts with "truth" values, only verifying no errors here
class LargeTurbopropFreighterBenchmark(unittest.TestCase):
    def build_and_run_problem(self, mission_method):
        if mission_method == 'energy':
            phase_info = deepcopy(energy_phase_info)

        elif mission_method == '2DOF':
            phase_info = deepcopy(two_dof_phase_info)

        # Build problem
        prob = AviaryProblem(verbosity=0)

        # load inputs from .csv to build engine
        options, _ = create_vehicle(
            'models/aircraft/large_turboprop_freighter/large_turboprop_freighter_GASP.csv'
        )
        options.set_val(Settings.PROBLEM_TYPE, 'SIZING')

        turboprop = TurbopropModel('turboprop', options=options)

        if mission_method == 'energy':
            options.set_val(Settings.EQUATIONS_OF_MOTION, 'height_energy')

        # load_inputs needs to be updated to accept an already existing aviary options
        prob.load_inputs(
            options,
            phase_info,
            engine_builders=[turboprop],
        )
        # if mission_method == 'energy':
        #     # energy method needs flight idle to converge in descent?
        #     prob.aviary_inputs.set_val(Aircraft.Engine.GENERATE_FLIGHT_IDLE, True)

        prob.check_and_preprocess_inputs()

        prob.build_model()
        prob.add_driver('IPOPT', max_iter=0, verbosity=0)
        prob.add_design_variables()
        prob.add_objective()
        prob.setup()

        prob.set_initial_guesses()
        prob.run_aviary_problem()

        return prob

    def test_bench_2DOF(self):
        prob = self.build_and_run_problem('2DOF')
        # TODO asserts

    @unittest.skip('Skipping due to convergence issues (possible drag too low in descent?)')
    def test_bench_energy(self):
        prob = self.build_and_run_problem('energy')
        # TODO asserts


if __name__ == '__main__':
    # unittest.main()
    test = LargeTurbopropFreighterBenchmark()
    # test.build_and_run_problem('2DOF')
    test.build_and_run_problem('energy')
