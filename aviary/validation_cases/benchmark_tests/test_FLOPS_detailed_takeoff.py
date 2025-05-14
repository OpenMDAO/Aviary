import copy
import unittest
import warnings

import dymos as dm
import openmdao.api as om
from openmdao.core.driver import Driver
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs

from aviary.models.N3CC.N3CC_data import inputs as _inputs
from aviary.models.N3CC.N3CC_data import (
    takeoff_liftoff_user_options as _takeoff_liftoff_user_options,
)
from aviary.models.N3CC.N3CC_data import takeoff_trajectory_builder as _takeoff_trajectory_builder
from aviary.subsystems.premission import CorePreMission
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.functions import set_aviary_initial_values, set_aviary_input_defaults
from aviary.utils.preprocessors import preprocess_options
from aviary.utils.test_utils.default_subsystems import get_default_mission_subsystems
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Dynamic


@use_tempdirs
class TestFLOPSDetailedTakeoff(unittest.TestCase):
    """Test detailed takeoff using N3CC data."""

    @require_pyoptsparse(optimizer='IPOPT')
    def bench_test_IPOPT(self):
        # raise unittest.SkipTest("IPOPT currently not working with this benchmark.")

        driver = om.pyOptSparseDriver()

        optimizer = 'IPOPT'
        driver.options['optimizer'] = optimizer

        driver.opt_settings['max_iter'] = 100
        driver.opt_settings['tol'] = 1e-3
        driver.opt_settings['print_level'] = 4
        # driver.opt_settings['hessian_approximation'] = 'limited-memory'
        # driver.opt_settings['nlp_scaling_method'] = None
        # driver.opt_settings['linear_solver'] = 'mumps'

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

        takeoff_trajectory_builder = copy.deepcopy(_takeoff_trajectory_builder)
        takeoff_liftoff_user_options = _takeoff_liftoff_user_options.deepcopy()

        takeoff = om.Problem()
        takeoff.driver = driver

        driver.declare_coloring()

        driver.add_recorder(om.SqliteRecorder(f'FLOPS_detailed_takeoff_traj_{optimizer}.sql'))

        driver.recording_options['record_derivatives'] = False

        engines = [build_engine_deck(aviary_options)]
        preprocess_options(aviary_options, engine_models=engines)

        default_premission_subsystems = get_default_mission_subsystems('FLOPS', engines)

        # Upstream static analysis for aero
        takeoff.model.add_subsystem(
            'pre_mission',
            CorePreMission(aviary_options=aviary_options, subsystems=default_premission_subsystems),
            promotes_inputs=['aircraft:*'],
            promotes_outputs=['aircraft:*', 'mission:*'],
        )

        # Instantiate the trajectory and add the phases
        traj = dm.Trajectory()
        takeoff.model.add_subsystem('traj', traj)

        takeoff_trajectory_builder.build_trajectory(
            aviary_options=aviary_options, model=takeoff.model, traj=traj
        )

        distance_max, units = takeoff_liftoff_user_options.get_item('distance_max')
        liftoff = takeoff_trajectory_builder.get_phase('takeoff_liftoff')

        liftoff.add_objective(Dynamic.Mission.DISTANCE, loc='final', ref=distance_max, units=units)

        # Insert a constraint for a fake decision speed, until abort is added.
        takeoff.model.add_constraint(
            'traj.takeoff_brake_release.states:velocity',
            equals=149.47,
            units='kn',
            ref=150.0,
            indices=[-1],
        )

        takeoff.model.add_constraint(
            'traj.takeoff_decision_speed.states:velocity',
            equals=155.36,
            units='kn',
            ref=159.0,
            indices=[-1],
        )

        varnames = [Aircraft.Wing.ASPECT_RATIO, Aircraft.Engine.SCALE_FACTOR]
        set_aviary_input_defaults(takeoff.model, varnames, aviary_options)

        setup_model_options(takeoff, aviary_options)

        # suppress warnings:
        # "input variable '...' promoted using '*' was already promoted using 'aircraft:*'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', om.PromotionWarning)
            takeoff.setup(check=True)

        set_aviary_initial_values(takeoff, aviary_options)

        # Turn off solver printing so that the SNOPT output is readable.
        takeoff.set_solver_print(level=0)

        takeoff_trajectory_builder.apply_initial_guesses(takeoff, 'traj')

        # run the problem
        dm.run_problem(takeoff, run_driver=True, simulate=True, make_plots=False)

        # liftoff.rhs_all.list_inputs(print_arrays=True)
        # takeoff.model.list_outputs(print_arrays=True)

        # Field length
        # N3CC FLOPS output line 2393
        desired = 5649.9  # ft

        actual = takeoff.model.get_val('traj.takeoff_liftoff.states:distance', units='ft')[-1]

        assert_near_equal(actual, desired, 2e-2)


if __name__ == '__main__':
    use_SNOPT = True

    z = TestFLOPSDetailedTakeoff()

    if use_SNOPT:
        z.bench_test_SNOPT()

    else:
        z.bench_test_IPOPT()
