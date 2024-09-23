import numpy as np
import unittest

from numpy.testing import assert_almost_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.subsystems.propulsion.turboprop_model import TurbopropModel
from aviary.utils.process_input_decks import create_vehicle
from aviary.variable_info.variables import Aircraft, Mission, Settings

from aviary.models.large_turboprop_freighter.phase_info import phase_info


@use_tempdirs
@unittest.skip("Skipping until all builders are updated with get_parameters()")
# TODO need to add asserts with "truth" values
class LargeTurbopropFreighterBenchmark(unittest.TestCase):

    def build_and_run_problem(self):

        # Build problem
        prob = AviaryProblem()

        # load inputs from .csv to build engine
        options, _ = create_vehicle(
            "models/large_turboprop_freighter/large_turboprop_freighter.csv"
        )

        turboprop = TurbopropModel('turboprop', options=options)

        # load_inputs needs to be updated to accept an already existing aviary options
        prob.load_inputs(
            "models/large_turboprop_freighter/large_turboprop_freighter.csv",
            phase_info,
            engine_builders=[turboprop],
        )

        # FLOPS aero specific stuff? Best guesses for values here
        # prob.aviary_inputs.set_val(Mission.Constraints.MAX_MACH, 0.5)
        # prob.aviary_inputs.set_val(Aircraft.Fuselage.AVG_DIAMETER, 4.125, 'm')

        prob.check_and_preprocess_inputs()
        prob.add_pre_mission_systems()
        prob.add_phases()
        prob.add_post_mission_systems()
        prob.link_phases()
        prob.add_driver("SLSQP", max_iter=0, verbosity=0)
        prob.add_design_variables()
        prob.add_objective()
        prob.setup()
        prob.set_initial_guesses()

        prob.run_aviary_problem("dymos_solution.db", make_plots=False)


if __name__ == '__main__':
    test = LargeTurbopropFreighterBenchmark()
    test.build_and_run_problem()
