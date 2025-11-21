import unittest

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs

from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.models.aircraft.large_turboprop_freighter.phase_info import energy_phase_info
from aviary.subsystems.propulsion.motor.motor_builder import MotorBuilder
from aviary.subsystems.propulsion.turboprop_model import TurbopropModel
from aviary.utils.process_input_decks import create_vehicle
from aviary.variable_info.variables import Aircraft, Mission, Settings


@use_tempdirs
# TODO need to add asserts with "truth" values
class LargeElectrifiedTurbopropFreighterBenchmark(unittest.TestCase):
    def build_and_run_problem(self):
        # Build problem
        prob = AviaryProblem(verbosity=0)

        # load inputs from .csv to build engine
        options, guesses = create_vehicle(
            'models/aircraft/large_turboprop_freighter/large_turboprop_freighter_GASP.csv'
        )

        options.set_val(Settings.EQUATIONS_OF_MOTION, 'height_energy')
        # options.set_val(Aircraft.Engine.NUM_ENGINES, 2)
        # options.set_val(Aircraft.Engine.WING_LOCATIONS, 0.385)
        scale_factor = 3
        options.set_val(Aircraft.Engine.RPM_DESIGN, 6000, 'rpm')
        options.set_val(Aircraft.Engine.FIXED_RPM, 6000, 'rpm')
        options.set_val(Aircraft.Engine.Gearbox.GEAR_RATIO, 5.88)
        options.set_val(Aircraft.Engine.Gearbox.EFFICIENCY, 1.0)
        options.set_val(Aircraft.Engine.SCALE_FACTOR, scale_factor)  # 11.87)
        options.set_val(
            Aircraft.Engine.SCALED_SLS_THRUST,
            options.get_val(Aircraft.Engine.REFERENCE_SLS_THRUST, 'lbf') * scale_factor,
            'lbf',
        )

        # turboprop = TurbopropModel('turboprop', options=options)
        # turboprop2 = TurbopropModel('turboprop2', options=options)

        motor = MotorBuilder(
            'motor',
        )

        electroprop = TurbopropModel('electroprop', options=options, shaft_power_model=motor)

        # load_inputs needs to be updated to accept an already existing aviary options
        prob.load_inputs(
            options,  # "models/aircraft/large_turboprop_freighter/large_turboprop_freighter_GASP.csv",
            energy_phase_info,
            engine_builders=[electroprop],
        )
        prob.aviary_inputs.set_val(Settings.VERBOSITY, 2)

        # FLOPS aero specific stuff? Best guesses for values here
        prob.aviary_inputs.set_val(Mission.Constraints.MAX_MACH, 0.5)
        prob.aviary_inputs.set_val(Aircraft.Wing.AREA, 1744.59, 'ft**2')
        # prob.aviary_inputs.set_val(Aircraft.Wing.ASPECT_RATIO, 10.078)
        prob.aviary_inputs.set_val(
            Aircraft.Wing.THICKNESS_TO_CHORD, 0.1500
        )  # average between root and chord T/C
        prob.aviary_inputs.set_val(Aircraft.Fuselage.MAX_WIDTH, 4.3, 'm')
        prob.aviary_inputs.set_val(Aircraft.Fuselage.MAX_HEIGHT, 3.95, 'm')
        prob.aviary_inputs.set_val(Aircraft.Fuselage.AVG_DIAMETER, 4.125, 'm')

        prob.check_and_preprocess_inputs()

        prob.build_model()
        prob.add_driver('IPOPT', max_iter=0, verbosity=0)
        prob.add_design_variables()
        prob.add_objective()

        prob.setup()
        # prob.model.list_vars(units=True, print_arrays=True)
        om.n2(prob)

        prob.run_aviary_problem()
        self.assertTrue(prob.result.success)

        om.n2(prob)


if __name__ == '__main__':
    test = LargeElectrifiedTurbopropFreighterBenchmark()
    test.build_and_run_problem()
