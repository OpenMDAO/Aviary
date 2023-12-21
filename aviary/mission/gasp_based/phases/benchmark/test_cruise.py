import unittest

import dymos as dm
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.mission.gasp_based.ode.breguet_cruise_ode import BreguetCruiseODESolution
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Dynamic


def make_cruise_problem():
    #
    # CRUISE UNDER BREGUET RANGE ASSUMPTION
    #
    # Initial States Fixed
    # Final States Free
    #
    # Controls:
    #   None
    #
    # Boundary Constraints:
    #   None
    #
    # Path Constraints:
    #   None
    #
    ode_args = dict(
        aviary_options=get_option_defaults(),
    )

    cruise = dm.AnalyticPhase(
        ode_class=BreguetCruiseODESolution,
        # Use the standard ode_args and update it for ground_roll dynamics
        ode_init_kwargs=ode_args,
        num_nodes=2
    )

    # Time here is really the independent variable through which we are integrating.
    # In the case of the Breguet Range ODE, it's mass.
    # We rely on mass being monotonically non-increasing across the phase.
    cruise.set_time_options(
        fix_initial=True,
        fix_duration=False,
        units="lbm",
        name="mass",
        duration_bounds=(-20000, 0),
        duration_ref=10000,
    )

    throttle_cruise = 0.930
    cruise.add_parameter(
        Dynamic.Mission.THROTTLE, opt=False, units="unitless", val=throttle_cruise, static_target=False
    )
    cruise.add_parameter(Dynamic.Mission.ALTITUDE, opt=False, val=37500.0, units='ft')
    cruise.add_parameter(Dynamic.Mission.MACH, opt=False, val=0.8, units="unitless")
    cruise.add_parameter("wing_area", opt=False, val=1370, units="ft**2")
    cruise.add_parameter("initial_cruise_range", opt=False, val=0.0, units="NM")
    cruise.add_parameter("initial_cruise_time", opt=False, val=0.0, units="s")

    cruise.add_timeseries_output("time", output_name="time", units="s")

    p = om.Problem()
    traj = p.model.add_subsystem("traj", dm.Trajectory())
    traj.add_phase("cruise", cruise)

    p.set_solver_print(level=-1)

    p.setup(force_alloc_complex=True)

    # SET TIME INITIAL GUESS
    p.set_val("traj.cruise.t_initial", 171481, units="lbm")  # Initial mass in cruise
    p.set_val("traj.cruise.t_duration", -10000, units="lbm")  # Mass of fuel consumed

    p.set_val("traj.cruise.parameters:altitude", val=37500.0, units="ft")
    p.set_val("traj.cruise.parameters:mach", val=0.8, units="unitless")
    p.set_val("traj.cruise.parameters:wing_area", val=1370, units="ft**2")
    p.set_val("traj.cruise.parameters:initial_cruise_range", val=0.0, units="NM")
    p.set_val("traj.cruise.parameters:initial_cruise_time", val=0.0, units="s")

    return p


@use_tempdirs
class TestCruise(unittest.TestCase):

    def assert_result(self, p):
        tf = p.get_val('traj.cruise.timeseries.states:cruise_time', units='s')[-1, 0]
        rf = p.get_val('traj.cruise.timeseries.states:distance', units='NM')[-1, 0]
        wf = p.get_val('traj.cruise.timeseries.mass', units='lbm')[-1, 0]

        print(f't_final: {tf:8.3f} s')
        print(f'w_final: {wf:8.3f} lbm')
        print(f'r_final: {rf:8.3f} NM')

        assert_near_equal(wf, 161481.0, tolerance=0.01)
        assert_near_equal(tf, 7368.56, tolerance=0.01)
        assert_near_equal(rf, 939.178, tolerance=0.01)

    @unittest.skip('Skipping this benchmark for now as the analytic cruise is not used in GASP currently.')
    def test_cruise_result(self):
        p = make_cruise_problem()
        dm.run_problem(p, run_driver=False, simulate=False)

        self.assert_result(p)
