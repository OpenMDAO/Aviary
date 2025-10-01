import unittest

from openmdao.utils.testing_utils import use_tempdirs

from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.models.aircraft.large_turboprop_freighter.phase_info import two_dof_phase_info
from aviary.subsystems.propulsion.turboprop_model import TurbopropModel
from aviary.utils.process_input_decks import create_vehicle
from aviary.variable_info.variables import Aircraft, Mission


@use_tempdirs
# TODO need to add asserts with "truth" values, only verifying no errors here
class LargeTurbopropFreighterBenchmark(unittest.TestCase):
    def build_and_run_problem(self):
        # Build problem
        prob = AviaryProblem(verbosity=0)

        # load inputs from .csv to build engine
        options, _ = create_vehicle(
            'models/aircraft/large_turboprop_freighter/large_turboprop_freighter_GASP.csv'
        )

        turboprop = TurbopropModel('turboprop', options=options)

        # load_inputs needs to be updated to accept an already existing aviary options
        prob.load_inputs(
            'models/aircraft/large_turboprop_freighter/large_turboprop_freighter_GASP.csv',
            two_dof_phase_info,
            engine_builders=[turboprop],
        )
        # FLOPS aero specific stuff? Best guesses for values here
        prob.aviary_inputs.set_val(Mission.Constraints.MAX_MACH, 0.5)
        prob.aviary_inputs.set_val(Aircraft.Fuselage.AVG_DIAMETER, 4.125, 'm')

        prob.check_and_preprocess_inputs()

        prob.build_model()
        prob.add_driver('IPOPT', max_iter=0, verbosity=0)
        prob.add_design_variables()
        prob.add_objective()
        prob.setup()
        prob.run_aviary_problem()

        self.assertTrue(prob.result.success)
        # om.n2(prob)


if __name__ == '__main__':
    test = LargeTurbopropFreighterBenchmark()
    test.build_and_run_problem()
