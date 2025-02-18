import numpy as np
import unittest
import openmdao.api as om


from numpy.testing import assert_almost_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.subsystems.propulsion.turboprop_model import TurbopropModel
from aviary.utils.process_input_decks import create_vehicle
from aviary.variable_info.variables import Aircraft, Mission, Settings

from aviary.models.large_turboprop_freighter.phase_info import (
    two_dof_phase_info,
    energy_phase_info,
)


@use_tempdirs
# TODO need to add asserts with "truth" values
class LargeTurbopropFreighterBenchmark(unittest.TestCase):

    def build_and_run_problem(self, mission_method):
        if mission_method == 'energy':
            phase_info = energy_phase_info

        elif mission_method == '2DOF':
            phase_info = two_dof_phase_info
        # Build problem
        prob = AviaryProblem()

        # load inputs from .csv to build engine
        options, _ = create_vehicle(
            "models/large_turboprop_freighter/large_turboprop_freighter_GASP.csv"
        )

        turboprop = TurbopropModel('turboprop', options=options)

        if mission_method == 'energy':
            options.set_val(Settings.EQUATIONS_OF_MOTION, 'height_energy')

        # load_inputs needs to be updated to accept an already existing aviary options
        prob.load_inputs(
            options,
            phase_info,
            engine_builders=[turboprop],
        )
        prob.aviary_inputs.set_val(Settings.VERBOSITY, 2)

        if mission_method == 'energy':
            # FLOPS aero specific stuff, best guesses for values here
            prob.aviary_inputs.set_val(Mission.Constraints.MAX_MACH, 0.5)
            prob.aviary_inputs.set_val(Aircraft.Fuselage.AVG_DIAMETER, 4.125, 'm')

        prob.check_and_preprocess_inputs()
        prob.add_pre_mission_systems()
        prob.add_phases()
        prob.add_post_mission_systems()
        prob.link_phases()
        prob.add_driver("IPOPT", max_iter=0, verbosity=0)
        prob.add_design_variables()
        prob.add_objective()
        prob.setup()

        prob.set_initial_guesses()
        prob.run_aviary_problem("dymos_solution.db")

        return prob

    def test_bench_2DOF(self):
        prob = self.build_and_run_problem('2DOF')
        # TODO asserts

    # @unittest.skip("Skipping until all builders are updated with get_parameters()")
    def test_bench_energy(self):
        prob = self.build_and_run_problem('energy')
        # TODO asserts


if __name__ == '__main__':
    # unittest.main()
    test = LargeTurbopropFreighterBenchmark()
    test.build_and_run_problem('energy')
