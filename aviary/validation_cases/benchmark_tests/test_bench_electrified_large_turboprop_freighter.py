import unittest
from copy import deepcopy

from numpy.testing import assert_almost_equal
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs

from aviary.core.aviary_problem import AviaryProblem
from aviary.models.aircraft.large_turboprop_freighter.electrified_phase_info import (
    energy_phase_info,
    two_dof_phase_info,
)
from aviary.subsystems.energy.battery_builder import BatteryBuilder
from aviary.subsystems.propulsion.motor.motor_builder import MotorBuilder
from aviary.subsystems.propulsion.turboprop_model import TurbopropModel
from aviary.utils.functions import get_path
from aviary.utils.process_input_decks import create_vehicle
from aviary.variable_info.variables import Aircraft, Dynamic, Mission, Settings


# @use_tempdirs
@require_pyoptsparse(optimizer='IPOPT')
# TODO need to add asserts with "truth" values
class LargeElectrifiedTurbopropFreighterBenchmark(unittest.TestCase):
    def build_and_run_problem(self, mission_method):
        if mission_method == 'energy':
            phase_info = deepcopy(energy_phase_info)

        elif mission_method == '2DOF':
            phase_info = deepcopy(two_dof_phase_info)

        # del phase_info['climb']
        # del phase_info['descent']

        # Build problem
        prob = AviaryProblem(verbosity=0)

        # load inputs from .csv to build engine
        options, _ = create_vehicle(
            'models/aircraft/large_turboprop_freighter/large_turboprop_freighter_GASP.csv'
        )
        options.set_val(Settings.PROBLEM_TYPE, 'SIZING')

        if mission_method == 'energy':
            options.set_val(Settings.EQUATIONS_OF_MOTION, 'energy_state')

        options.set_val(Aircraft.CrewPayload.CARGO_MASS, 0, 'lbm')

        # set up electric propulsion
        # TODO make separate input file for electroprop freighter?
        options.set_val(Aircraft.Engine.SCALE_FACTOR, 1)
        options.set_val(Aircraft.Engine.RPM_DESIGN, 6_000, 'rpm')  # max RPM of motor map
        options.delete(Aircraft.Engine.FIXED_RPM)
        # options.set_val(Aircraft.Engine.FIXED_RPM, 6000, 'rpm')
        # match propeller RPM of gas turboprop
        options.set_val(Aircraft.Engine.Gearbox.GEAR_RATIO, 5.88)
        options.set_val(Aircraft.Engine.Gearbox.EFFICIENCY, 1.0)
        options.set_val(Aircraft.Battery.PACK_ENERGY_DENSITY, 1_000, 'W*h/kg')

        options.set_val(
            Aircraft.Engine.Motor.DATA_FILE, get_path('electric_motor_1800Nm_6000rpm.csv')
        )

        motor = MotorBuilder(
            name='motor',
        )

        electroprop = TurbopropModel('electroprop', options=options, shaft_power_model=motor)

        # load_inputs needs to be updated to accept an already existing aviary options
        prob.load_inputs(
            options,  # "models/aircraft/large_turboprop_freighter/large_turboprop_freighter_GASP.csv",
            phase_info,
        )
        prob.load_external_subsystems([electroprop])

        prob.load_external_subsystems([BatteryBuilder()])

        prob.check_and_preprocess_inputs()

        prob.build_model()

        prob.add_driver('SNOPT', max_iter=100, verbosity=1)
        prob.add_design_variables()
        prob.model.add_design_var(
            Aircraft.Engine.SCALE_FACTOR,
            units='unitless',
            lower=0.25,
            upper=2,
            ref=1,
        )
        prob.model.add_design_var(
            Aircraft.Battery.PACK_MASS, units='lbm', lower=10_000, upper=75_000, ref=10_000
        )

        final_phase_name = prob.model.regular_phases[-1]
        prob.model.add_objective(
            f'traj.{final_phase_name}.timeseries.cumulative_electric_energy_used',
            index=-1,
            ref=1e6,
        )
        # prob.add_objective('mass')

        prob.setup()

        # initial guess for pack mass.
        prob.set_val(Aircraft.Battery.PACK_MASS, val=25_000.0, units='lbm')

        prob.run_aviary_problem()

        # prob.model.list_vars(units=True, print_arrays=True)
        return prob

    @unittest.skip('Skipping until subsystems with states can be used in 2DOF cruise')
    def test_bench_2DOF(self):
        prob = self.build_and_run_problem('2DOF')
        self.assertTrue(prob.result.success)
        # TODO asserts

    @unittest.skip('Skipping due to convergence issues (possible drag too low in descent?)')
    def test_bench_energy(self):
        prob = self.build_and_run_problem('energy')
        prob.model.list_vars()
        self.assertTrue(prob.result.success)
        # TODO asserts


if __name__ == '__main__':
    # unittest.main()
    test = LargeElectrifiedTurbopropFreighterBenchmark()
    test.build_and_run_problem('energy')
