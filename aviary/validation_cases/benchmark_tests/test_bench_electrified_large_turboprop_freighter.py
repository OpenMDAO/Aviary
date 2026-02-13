import unittest
from copy import deepcopy

from numpy.testing import assert_almost_equal
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs

from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.models.aircraft.large_turboprop_freighter.electrified_phase_info import (
    energy_phase_info,
    two_dof_phase_info,
)
from aviary.subsystems.energy.battery_builder import BatteryBuilder
from aviary.subsystems.propulsion.motor.motor_builder import MotorBuilder
from aviary.subsystems.propulsion.turboprop_model import TurbopropModel
from aviary.utils.process_input_decks import create_vehicle
from aviary.variable_info.variables import Aircraft, Mission, Settings


@use_tempdirs
@require_pyoptsparse(optimizer='IPOPT')
# TODO need to add asserts with "truth" values
class LargeElectrifiedTurbopropFreighterBenchmark(unittest.TestCase):
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

        if mission_method == 'energy':
            options.set_val(Settings.EQUATIONS_OF_MOTION, 'height_energy')

        # set up electric propulsion
        # TODO make separate input file for electroprop freighter?
        scale_factor = 17.77  # target is ~32 kN*m torque
        options.set_val(Aircraft.Engine.RPM_DESIGN, 6000, 'rpm')  # max RPM of motor map
        options.set_val(Aircraft.Engine.FIXED_RPM, 6000, 'rpm')
        # match propeller RPM of gas turboprop
        options.set_val(Aircraft.Engine.Gearbox.GEAR_RATIO, 5.88)
        options.set_val(Aircraft.Engine.Gearbox.EFFICIENCY, 1.0)
        options.set_val(Aircraft.Engine.SCALE_FACTOR, scale_factor)  # 11.87)
        options.set_val(
            Aircraft.Engine.SCALED_SLS_THRUST,
            options.get_val(Aircraft.Engine.REFERENCE_SLS_THRUST, 'lbf') * scale_factor,
            'lbf',
        )
        options.set_val(Aircraft.Battery.PACK_ENERGY_DENSITY, 1000, 'kW*h/kg')

        motor = MotorBuilder(
            'motor',
        )

        electroprop = TurbopropModel('electroprop', options=options, shaft_power_model=motor)

        # load_inputs needs to be updated to accept an already existing aviary options
        prob.load_inputs(
            options,  # "models/aircraft/large_turboprop_freighter/large_turboprop_freighter_GASP.csv",
            phase_info,
        )
        prob.load_external_subsystems([electroprop])

        prob.aviary_inputs.set_val(Settings.VERBOSITY, 0)

        # if mission_method == 'energy':
        #     # FLOPS aero specific stuff? Best guesses for values here
        #     prob.aviary_inputs.set_val(Mission.Constraints.MAX_MACH, 0.5)
        #     prob.aviary_inputs.set_val(Aircraft.Wing.AREA, 1744.59, 'ft**2')
        #     # prob.aviary_inputs.set_val(Aircraft.Wing.ASPECT_RATIO, 10.078)
        #     prob.aviary_inputs.set_val(
        #         Aircraft.Wing.THICKNESS_TO_CHORD, 0.1500
        #     )  # average between root and chord T/C
        #     prob.aviary_inputs.set_val(Aircraft.Fuselage.MAX_WIDTH, 4.3, 'm')
        #     prob.aviary_inputs.set_val(Aircraft.Fuselage.MAX_HEIGHT, 3.95, 'm')
        #     prob.aviary_inputs.set_val(Aircraft.Fuselage.REF_DIAMETER, 4.125, 'm')

        prob.load_external_subsystems([BatteryBuilder()])

        prob.check_and_preprocess_inputs()

        prob.build_model()
        prob.add_driver('IPOPT', max_iter=0, verbosity=0)
        prob.add_design_variables()
        # prob.model.add_design_var(
        #     Aircraft.Engine.SCALE_FACTOR,
        #     units='unitless',
        #     lower=1,
        #     upper=25,
        #     ref=10,
        # )
        prob.add_objective()

        prob.setup()

        prob.run_aviary_problem()
        self.assertTrue(prob.result.success)

    @unittest.skip('Skipping until subsystems with states can be used in 2DOF cruise')
    def test_bench_2DOF(self):
        prob = self.build_and_run_problem('2DOF')
        # TODO asserts

    @unittest.skip('Skipping due to convergence issues (possible drag too low in descent?)')
    def test_bench_energy(self):
        prob = self.build_and_run_problem('energy')
        # TODO asserts


if __name__ == '__main__':
    # unittest.main()
    test = LargeElectrifiedTurbopropFreighterBenchmark()
    test.build_and_run_problem('2DOF')
