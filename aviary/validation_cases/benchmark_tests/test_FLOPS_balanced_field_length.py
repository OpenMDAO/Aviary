import copy
import unittest
import warnings

import dymos as dm
import openmdao.api as om
from openmdao.core.driver import Driver
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs

from aviary.interface.methods_for_level2 import AviaryProblem

from aviary.utils.functions import set_aviary_initial_values
from aviary.utils.preprocessors import preprocess_crewpayload
from aviary.models.N3CC.N3CC_data import \
    balanced_liftoff_user_options as _takeoff_liftoff_user_options
from aviary.models.N3CC.N3CC_data import \
    balanced_trajectory_builder as _takeoff_trajectory_builder
from aviary.models.N3CC.N3CC_data import \
    inputs as _inputs
from aviary.variable_info.variables import Dynamic as _Dynamic
from aviary.variable_info.variables_in import VariablesIn
from aviary.interface.default_phase_info.height_energy import default_premission_subsystems
from aviary.subsystems.premission import CorePreMission

Dynamic = _Dynamic.Mission


@use_tempdirs
class TestFLOPSBalancedFieldLength(unittest.TestCase):
    @require_pyoptsparse(optimizer='IPOPT')
    def bench_test_IPOPT(self):
        # raise unittest.SkipTest("IPOPT currently not working with this benchmark.")

        driver = om.pyOptSparseDriver()

        optimizer = 'IPOPT'
        driver.options['optimizer'] = optimizer

        driver.opt_settings['max_iter'] = 100
        driver.opt_settings['tol'] = 1e-3
        driver.opt_settings['print_level'] = 4

        self._do_run(driver, optimizer)

    @require_pyoptsparse(optimizer='SNOPT')
    def bench_test_SNOPT(self):
        driver = om.pyOptSparseDriver()

        optimizer = 'SNOPT'
        driver.options['optimizer'] = optimizer

        driver.opt_settings['Major iterations limit'] = 50
        driver.opt_settings['Major optimality tolerance'] = 1e-4
        driver.opt_settings['Major feasibility tolerance'] = 1e-6
        driver.opt_settings['iSumm'] = 6

        self._do_run(driver, optimizer)

    def _do_run(self, driver: Driver, optimizer, *args):
        aviary_options = _inputs.deepcopy()
        preprocess_crewpayload(aviary_options)

        takeoff_trajectory_builder = copy.deepcopy(_takeoff_trajectory_builder)
        takeoff_liftoff_user_options = _takeoff_liftoff_user_options.deepcopy()

        takeoff = om.Problem()
        takeoff.driver = driver

        driver.declare_coloring()

        driver.add_recorder(om.SqliteRecorder(
            f'FLOPS_detailed_takeoff_traj_{optimizer}.sql'))

        driver.recording_options['record_derivatives'] = False

        # Upstream static analysis for aero
        takeoff.model.add_subsystem(
            'core_subsystems',
            CorePreMission(
                aviary_options=aviary_options,
                subsystems=default_premission_subsystems,
            ),
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=['aircraft:*', 'mission:*'])

        # Instantiate the trajectory and add the phases
        traj = dm.Trajectory()
        takeoff.model.add_subsystem('traj', traj)

        takeoff_trajectory_builder.build_trajectory(
            aviary_options=aviary_options, model=takeoff.model, traj=traj)

        max_range, units = takeoff_liftoff_user_options.get_item('max_range')
        liftoff = takeoff_trajectory_builder.get_phase('balanced_liftoff')

        liftoff.add_objective(
            Dynamic.RANGE, loc='final', ref=max_range, units=units)

        takeoff.model.add_subsystem(
            'input_sink',
            VariablesIn(aviary_options=aviary_options),
            promotes_inputs=['*'],
            promotes_outputs=['*']
        )

        # suppress warnings:
        # "input variable '...' promoted using '*' was already promoted using 'aircraft:*'
        with warnings.catch_warnings():
            # Set initial default values for all aircraft variables.
            set_aviary_initial_values(takeoff.model, aviary_options)

            warnings.simplefilter("ignore", om.PromotionWarning)
            takeoff.setup(check=True)

        # Turn off solver printing so that the SNOPT output is readable.
        takeoff.set_solver_print(level=0)

        takeoff_trajectory_builder.apply_initial_guesses(takeoff, 'traj')

        # run the problem
        dm.run_problem(takeoff, run_driver=True, simulate=True, make_plots=False)

        # takeoff.model.traj.phases.brake_release_to_decision_speed.list_inputs(print_arrays=True)
        # takeoff.model.list_outputs(print_arrays=True)

        # Field Length
        # N3CC FLOPS output Line 2282
        desired = 7032.65
        actual = takeoff.model.get_val(
            'traj.balanced_liftoff.states:range', units='ft')[-1]
        assert_near_equal(actual, desired, 2e-2)

        # Decision Time
        # N3CC FLOPS output Line 2287
        desired = 29.52
        actual = takeoff.model.get_val(
            'traj.balanced_brake_release.t', units='s')[-1]
        assert_near_equal(actual, desired, 2e-2)

        # Liftoff Time
        # N3CC FLOPS output Line 2289
        desired = 36.63
        actual = takeoff.model.get_val(
            'traj.balanced_rotate.t', units='s')[-1]
        assert_near_equal(actual, desired, .05)

        # Rotation Speed
        # N3CC FLOPS output Line 2289
        desired = 156.55
        actual = takeoff.model.get_val(
            'traj.balanced_rotate.states:velocity', units='kn')[-1]
        assert_near_equal(actual, desired, 2e-2)


if __name__ == '__main__':
    use_SNOPT = True

    z = TestFLOPSBalancedFieldLength()

    if use_SNOPT:
        z.bench_test_SNOPT()

    else:
        z.bench_test_IPOPT()
