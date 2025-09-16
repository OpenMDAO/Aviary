import copy
import unittest
import warnings

import dymos as dm
import openmdao.api as om
from openmdao.core.driver import Driver
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs

from aviary.models.aircraft.advanced_single_aisle.advanced_single_aisle_data import (
    inputs as _inputs,
)
from aviary.models.aircraft.advanced_single_aisle.advanced_single_aisle_data import (
    landing_fullstop_user_options as _landing_fullstop_user_options,
)
from aviary.models.aircraft.advanced_single_aisle.advanced_single_aisle_data import (
    landing_trajectory_builder as _landing_trajectory_builder,
)
from aviary.subsystems.premission import CorePreMission
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.functions import set_aviary_initial_values, set_aviary_input_defaults
from aviary.utils.preprocessors import preprocess_options
from aviary.utils.test_utils.default_subsystems import get_default_mission_subsystems
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Dynamic


@use_tempdirs
class TestFLOPSDetailedLanding(unittest.TestCase):
    """Test detailed landing using N3CC data."""

    @require_pyoptsparse(optimizer='IPOPT')
    def bench_test_IPOPT(self):
        driver = om.pyOptSparseDriver()

        optimizer = 'IPOPT'
        driver.options['optimizer'] = optimizer

        driver.opt_settings['max_iter'] = 100
        driver.opt_settings['tol'] = 1.0e-6
        driver.opt_settings['print_level'] = 4
        driver.opt_settings['mu_init'] = 1e-5

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

        landing_trajectory_builder = copy.deepcopy(_landing_trajectory_builder)
        landing_fullstop_user_options = _landing_fullstop_user_options.deepcopy()

        landing = om.Problem()
        landing.driver = driver

        driver.add_recorder(om.SqliteRecorder(f'FLOPS_detailed_landing_traj_{optimizer}.sql'))

        driver.recording_options['record_derivatives'] = False

        engines = [build_engine_deck(aviary_options)]
        preprocess_options(aviary_options, engine_models=engines)

        default_premission_subsystems = get_default_mission_subsystems('FLOPS', engines)

        # Upstream static analysis for aero
        landing.model.add_subsystem(
            'pre_mission',
            CorePreMission(aviary_options=aviary_options, subsystems=default_premission_subsystems),
            promotes_inputs=['aircraft:*'],
            promotes_outputs=['aircraft:*', 'mission:*'],
        )

        # Instantiate the trajectory and add the phases
        traj = dm.Trajectory()
        landing.model.add_subsystem('traj', traj)

        landing_trajectory_builder.build_trajectory(
            aviary_options=aviary_options, model=landing.model, traj=traj
        )

        distance_max, units = landing_fullstop_user_options.get_item('distance_max')
        fullstop = landing_trajectory_builder.get_phase('landing_fullstop')

        fullstop.add_objective(Dynamic.Mission.DISTANCE, loc='final', ref=distance_max, units=units)

        varnames = [Aircraft.Wing.ASPECT_RATIO]
        set_aviary_input_defaults(landing.model, varnames, aviary_options)

        setup_model_options(landing, aviary_options)

        # suppress warnings:
        # "input variable '...' promoted using '*' was already promoted using 'aircraft:*'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', om.PromotionWarning)
            landing.setup(check=True)

        set_aviary_initial_values(landing, aviary_options)

        # Turn off solver printing so that the SNOPT output is readable.
        landing.set_solver_print(level=0)

        landing_trajectory_builder.apply_initial_guesses(landing, 'traj')

        # run the problem
        landing.result = dm.run_problem(landing, run_driver=True, simulate=True, make_plots=False)

        # self.assertTrue(landing.result.success)

        # Field length
        # N3CC FLOPS output line 1773
        # base = -954.08 # ft
        # N3CC FLOPS output line 1842
        desired = 3409.47  # ft

        actual = landing.model.get_val('traj.landing_fullstop.states:distance', units='ft')[-1]

        assert_near_equal(actual, desired, 0.05)

        # TOUCHDOWN time
        # N3CC FLOPS output line 1773
        # base = -4.08 # s
        # N3CC FLOPS output line 1849
        desired = 4.22  # s

        actual = landing.model.get_val('traj.landing_flare.t', units='s')[-1]

        assert_near_equal(actual, desired, 0.10)

        # End of landing time
        # N3CC FLOPS output line 1852
        desired = 24.49  # s

        actual = landing.model.get_val('traj.landing_fullstop.t', units='s')[-1]

        assert_near_equal(actual, desired, 0.05)


if __name__ == '__main__':
    use_SNOPT = False

    z = TestFLOPSDetailedLanding()

    if use_SNOPT:
        z.bench_test_SNOPT()

    else:
        z.bench_test_IPOPT()
