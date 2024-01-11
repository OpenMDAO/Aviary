import copy
import unittest
import warnings

import dymos as dm
import openmdao.api as om
from openmdao.core.driver import Driver
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs

from aviary.subsystems.premission import CorePreMission

from aviary.utils.functions import set_aviary_initial_values

from aviary.models.N3CC.N3CC_data import (
    inputs as _inputs, outputs as _outputs,
    landing_trajectory_builder as _landing_trajectory_builder,
    landing_fullstop_user_options as _landing_fullstop_user_options)

from aviary.variable_info.variables import Dynamic as _Dynamic
from aviary.interface.default_phase_info.height_energy import default_premission_subsystems
from aviary.utils.preprocessors import preprocess_crewpayload
from aviary.variable_info.variables_in import VariablesIn


Dynamic = _Dynamic.Mission


@use_tempdirs
class TestFLOPSDetailedLanding(unittest.TestCase):
    # @require_pyoptsparse(optimizer='IPOPT')
    # def bench_test_IPOPT(self):
    #     driver = om.pyOptSparseDriver()

    #     optimizer = 'IPOPT'
    #     driver.options['optimizer'] = optimizer

    #     driver.opt_settings['max_iter'] = 100
    #     driver.opt_settings['tol'] = 1.0E-6
    #     driver.opt_settings['print_level'] = 4
    #     driver.opt_settings['mu_init'] = 1e-5

    #     self._do_run(driver, optimizer)

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

        driver.add_recorder(om.SqliteRecorder(
            f'FLOPS_detailed_landing_traj_{optimizer}.sql'))

        driver.recording_options['record_derivatives'] = False

        preprocess_crewpayload(aviary_options)

        # Upstream static analysis for aero
        landing.model.add_subsystem(
            'pre_mission',
            CorePreMission(aviary_options=aviary_options,
                           subsystems=default_premission_subsystems),
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=['aircraft:*', 'mission:*'])

        # Instantiate the trajectory and add the phases
        traj = dm.Trajectory()
        landing.model.add_subsystem('traj', traj)

        landing_trajectory_builder.build_trajectory(
            aviary_options=aviary_options, model=landing.model, traj=traj)

        max_range, units = landing_fullstop_user_options.get_item('max_range')
        fullstop = landing_trajectory_builder.get_phase('landing_fullstop')

        fullstop.add_objective(Dynamic.RANGE, loc='final', ref=max_range, units=units)

        landing.model.add_subsystem(
            'input_sink',
            VariablesIn(aviary_options=aviary_options),
            promotes_inputs=['*'],
            promotes_outputs=['*']
        )

        # suppress warnings:
        # "input variable '...' promoted using '*' was already promoted using 'aircraft:*'
        with warnings.catch_warnings():
            # Set initial default values for all aircraft variables.
            set_aviary_initial_values(landing.model, aviary_options)

            warnings.simplefilter("ignore", om.PromotionWarning)
            landing.setup(check=True)

        # Turn off solver printing so that the SNOPT output is readable.
        landing.set_solver_print(level=0)

        landing_trajectory_builder.apply_initial_guesses(landing, 'traj')

        # run the problem
        dm.run_problem(landing, run_driver=True, simulate=True, make_plots=False)

        # Field length
        # N3CC FLOPS output line 1773
        base = -954.08  # ft
        # N3CC FLOPS output line 1842
        desired = 3409.47  # ft

        actual = landing.model.get_val(
            'traj.landing_fullstop.states:range', units='ft')[-1]

        assert_near_equal(actual, desired, 0.05)

        # TOUCHDOWN time
        # N3CC FLOPS output line 1773
        base = -4.08  # s
        # N3CC FLOPS output line 1849
        desired = 4.22  # s

        actual = landing.model.get_val(
            'traj.landing_flare.t', units='s')[-1]

        assert_near_equal(actual, desired, 0.10)

        # End of landing time
        # N3CC FLOPS output line 1852
        desired = 24.49  # s

        actual = landing.model.get_val(
            'traj.landing_fullstop.t', units='s')[-1]

        assert_near_equal(actual, desired, 0.05)


if __name__ == '__main__':
    use_SNOPT = True

    z = TestFLOPSDetailedLanding()

    if use_SNOPT:
        z.bench_test_SNOPT()

    else:
        z.bench_test_IPOPT()
