"""
Top level run script for an Aviary problem.

This theoretically could be the Level 2 (intermediate) user's
entry point to Aviary.
"""

from aviary.api import AviaryProblem
from aviary.api import AnalysisScheme, SpeedType, AlphaModes, Verbosity
from aviary.api import FlexibleTraj
from aviary.api import SGMCruise, SGMDescent
from aviary.api import Dynamic


def custom_run_aviary(aircraft_filename, optimizer=None,
                      analysis_scheme=AnalysisScheme.COLLOCATION, objective_type=None,
                      record_filename='dymos_solution.db', restart_filename=None, max_iter=50,
                      run_driver=True, make_plots=True, phase_info_parameterization=None,
                      optimization_history_filename=None, verbosity=Verbosity.BRIEF):
    """
    This function runs the aviary optimization problem for the specified aircraft configuration and mission.

    It first creates an instance of the AviaryProblem class using the given mission_method and mass_method.
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

    from aviary.interface.default_phase_info.two_dof_fiti import ascent_phases, \
        add_default_sgm_args, phase_info_parameterization

    phase_info = {
        **ascent_phases,
        'cruise': {
            'kwargs': dict(
                input_speed_type=SpeedType.MACH,
                input_speed_units="unitless",
                alpha_mode=AlphaModes.REQUIRED_LIFT,
            ),
            'builder': SGMCruise,
            'user_options': {
                'mach': (0, 'unitless'),
                'attr:distance_trigger': (500, 'NM'),
            },
        },
        'desc1': {
            'kwargs': dict(
                input_speed_type=SpeedType.MACH,
                input_speed_units='unitless',
                speed_trigger_units='kn',
            ),
            'builder': SGMDescent,
            'user_options': {
                'alt_trigger': (10000, 'ft'),
                'mach': (0, 'unitless'),
                'speed_trigger': (350, 'kn'),
                Dynamic.Mission.THROTTLE: (0, 'unitless'),
            },
            'descent_phase': True,
        },
    }

    # Load aircraft and options data from user
    # Allow for user overrides here
    prob.load_inputs(aircraft_filename, phase_info)

    # Preprocess inputs
    prob.check_and_preprocess_inputs()

    prob.add_pre_mission_systems()

    # ######################################## #
    # replace the default trajectory with a custom trajectory
    # This trajectory uses the full GASP based ascent profile,
    # a distance based cruise, and a simplified descent
    phase_info, _ = phase_info_parameterization(phase_info, None, prob.aviary_inputs)
    add_default_sgm_args(phase_info, prob.ode_args)
    traj = FlexibleTraj(
        Phases=phase_info,
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
            ('groundroll', Dynamic.Mission.VELOCITY, 0,),
            ('climb3', Dynamic.Mission.ALTITUDE, 0,),
            ('cruise', Dynamic.Mission.DISTANCE, 0,),
        ],
    )
    prob.traj = prob.model.add_subsystem('traj', traj)
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

    prob.run_aviary_problem(
        record_filename, restart_filename=restart_filename, run_driver=run_driver, make_plots=make_plots)

    return prob


if __name__ == "__main__":
    input_deck = 'models/large_single_aisle_1/large_single_aisle_1_GwGm.csv'
    custom_run_aviary(
        input_deck, analysis_scheme=AnalysisScheme.SHOOTING, run_driver=False)
