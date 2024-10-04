import numpy as np
import unittest
import openmdao.api as om


from numpy.testing import assert_almost_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.subsystems.propulsion.turboprop_model import TurbopropModel
from aviary.subsystems.propulsion.motor.motor_builder import MotorBuilder
from aviary.utils.process_input_decks import create_vehicle
from aviary.variable_info.variables import Aircraft, Mission, Settings

from aviary.models.large_turboprop_freighter.phase_info import (
    two_dof_phase_info,
    energy_phase_info,
)


@use_tempdirs
# TODO need to add asserts with "truth" values
class LargeTurbopropFreighterBenchmark(unittest.TestCase):

    def build_and_run_problem(self):

        # Build problem
        prob = AviaryProblem()

        # load inputs from .csv to build engine
        options, guesses = create_vehicle(
            "models/large_turboprop_freighter/large_turboprop_freighter.csv"
        )

        # options.set_val(Aircraft.Engine.NUM_ENGINES, 2)
        # options.set_val(Aircraft.Engine.WING_LOCATIONS, 0.385)
        options.set_val(Aircraft.Engine.RPM_DESIGN, 1_019.916, 'rpm')
        options.set_val(Aircraft.Engine.Gearbox.GEAR_RATIO, 1.0)

        # turboprop = TurbopropModel('turboprop', options=options)
        # turboprop2 = TurbopropModel('turboprop2', options=options)

        motor = MotorBuilder(
            'motor',
        )

        electroprop = TurbopropModel(
            'electroprop', options=options, shaft_power_model=motor
        )

        # load_inputs needs to be updated to accept an already existing aviary options
        prob.load_inputs(
            options,  # "models/large_turboprop_freighter/large_turboprop_freighter.csv",
            two_dof_phase_info,
            engine_builders=[electroprop],
        )
        prob.aviary_inputs.set_val(Settings.VERBOSITY, 2)

        # FLOPS aero specific stuff? Best guesses for values here
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
        # prob.model.list_vars(units=True, print_arrays=True)
        # om.n2(prob)

        prob.set_initial_guesses()
        prob.run_aviary_problem("dymos_solution.db")

        om.n2(prob)


if __name__ == '__main__':
    test = LargeTurbopropFreighterBenchmark()
    test.build_and_run_problem()
