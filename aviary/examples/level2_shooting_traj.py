"""
Top level run script for an Aviary problem.

This theoretically could be the Level 2 (intermediate) user's
entry point to Aviary.
"""

from aviary.api import AviaryProblem
from aviary.api import AnalysisScheme, SpeedType, AlphaModes
from aviary.api import FlexibleTraj
from aviary.api import create_2dof_based_ascent_phases
from aviary.api import SGMCruise, SGMDescent
from aviary.api import Dynamic


def run_aviary(aircraft_filename, phase_info, optimizer=None, analysis_scheme=AnalysisScheme.COLLOCATION,
               objective_type=None, record_filename='dymos_solution.db', restart_filename=None, max_iter=50, run_driver=True, make_plots=True):
    """
    This function runs the aviary optimization problem for the specified aircraft configuration and mission.

    It first creates an instance of the AviaryProblem class using the given phase_info, mission_method, and mass_method.
    It then loads aircraft and options data from the user-provided aircraft_filename and checks for any clashing inputs.
    Pre-mission systems are added, phases are added and linked, then post-mission systems are added.
    A driver is added using the specified optimizer (The default optimizer depends on the analysis scheme).
    Design variables and the optimization objective are added, and the problem is set up.
    Initial guesses are set and the optimization problem is run.
    The function returns the AviaryProblem instance.

    A user can modify these methods or add their own to modify the behavior
    of the Aviary problem.
    """
    # Build problem
    prob = AviaryProblem(analysis_scheme)

    # Load aircraft and options data from user
    # Allow for user overrides here
    prob.load_inputs(aircraft_filename, phase_info)


# Preprocess inputs
    prob.check_and_preprocess_inputs()

    prob.add_pre_mission_systems()

# ######################################## #
# replace the default trajectory with a custom trajectory
# This trajectory uses the full GASP based ascent profile,
# a Breguet cruise, and a simplified descent
    ascent_phases = create_2dof_based_ascent_phases(
        prob.ode_args,
        cruise_alt=prob.cruise_alt,
        cruise_mach=prob.cruise_mach)

    cruise_kwargs = dict(
        input_speed_type=SpeedType.MACH,
        input_speed_units="unitless",
        ode_args=prob.ode_args,
        alpha_mode=AlphaModes.REQUIRED_LIFT,
        simupy_args=dict(
            DEBUG=True,
            blocked_state_names=['engine.nox', 'nox',
                                 'TAS', Dynamic.Mission.FLIGHT_PATH_ANGLE],
        ),
    )
    cruise_vals = {
        'mach': {'val': prob.cruise_mach, 'units': cruise_kwargs['input_speed_units']},
        'distance_trigger': {'val': 300, 'units': 'NM'},
    }

    descent1_kwargs = dict(
        input_speed_type=SpeedType.MACH,
        input_speed_units="unitless",
        speed_trigger_units='kn',
        ode_args=prob.ode_args,
        simupy_args=dict(
            DEBUG=False,
            blocked_state_names=['engine.nox', 'nox'],
        ),
    )
    descent1_vals = {
        'alt_trigger': {'val': 10000, 'units': 'ft'},
        'mach': {'val': prob.cruise_mach, 'units': None},
        'speed_trigger': {'val': 350, 'units': 'kn'}}

    phases = {
        **ascent_phases,
        'cruise': {
            'ode': SGMCruise(**cruise_kwargs),
            'vals_to_set': cruise_vals,
        },
        'descent1': {
            'ode': SGMDescent(**descent1_kwargs),
            'vals_to_set': descent1_vals
        },
    }

    traj = FlexibleTraj(
        Phases=phases,
        traj_final_state_output=[
            Dynamic.Mission.MASS,
            Dynamic.Mission.DISTANCE,
        ],
        traj_initial_state_input=[
            Dynamic.Mission.MASS,
            Dynamic.Mission.DISTANCE,
            Dynamic.Mission.ALTITUDE,
        ],
        traj_event_trigger_input=[
            (phases['groundroll']['ode'], "TAS", 0,),
            (phases['climb3']['ode'], Dynamic.Mission.ALTITUDE, 0,),
            (phases['cruise']['ode'], Dynamic.Mission.MASS, 0,),
        ],
    )
    traj = prob.model.add_subsystem('traj', traj)

    prob.traj = traj
# ######################################## #

    prob.add_post_mission_systems()

    # Link phases and variables
    prob.link_phases()

    prob.add_driver(optimizer, max_iter=max_iter)

    prob.add_design_variables()

    # Load optimization problem formulation
    # Detail which variables the optimizer can control
    prob.add_objective(objective_type=objective_type)

    prob.setup()

    prob.set_initial_guesses()

    prob.failed = prob.run_aviary_problem(
        record_filename, restart_filename=restart_filename, run_driver=run_driver, make_plots=make_plots)

    return prob


if __name__ == "__main__":
    from aviary.interface.default_phase_info.two_dof import phase_info
    input_deck = 'models/large_single_aisle_1/large_single_aisle_1_GwGm.csv'
    run_aviary(input_deck, phase_info,
               analysis_scheme=AnalysisScheme.SHOOTING, run_driver=False)
