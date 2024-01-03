"""
This file contains functions needed to run Aviary using the Level 1 interface.
"""
import os
import importlib.util
import sys

import openmdao.api as om
from aviary.variable_info.enums import AnalysisScheme
from aviary.interface.methods_for_level2 import AviaryProblem


def run_aviary(aircraft_filename, phase_info, mission_method, mass_method, optimizer=None,
               analysis_scheme=AnalysisScheme.COLLOCATION, objective_type=None,
               record_filename='dymos_solution.db', restart_filename=None, max_iter=50,
               run_driver=True, make_plots=True, phase_info_parameterization=None,
               optimization_history_filename=None):
    """
    Run the Aviary optimization problem for a specified aircraft configuration and mission.

    This function creates an instance of the AviaryProblem class using provided phase information,
    mission, and mass methods. It processes aircraft and options data from the given aircraft filename,
    checks for clashing inputs, and sets up pre- and post-mission systems. The optimization problem is formulated,
    initialized with guesses, and run to find a solution.

    Parameters
    ----------
    aircraft_filename : str
        Filename from which to load the aircraft and options data.
    phase_info : dict
        Information about the phases of the mission.
    mission_method : str
        The method used for defining the mission; can be 'GASP', 'FLOPS', 'solved', or 'simple'.
    mass_method : str
        The method used for calculating the mass; can be 'GASP' or 'FLOPS'.
    optimizer : str
        The optimizer to use.
    analysis_scheme : AnalysisScheme, optional
        The analysis scheme to use, defaults to AnalysisScheme.COLLOCATION.
    objective_type : str, optional
        Type of the optimization objective.
    record_filename : str, optional
        Filename for recording the solution, defaults to 'dymos_solution.db'.
    restart_filename : str, optional
        Filename to use for restarting the optimization, if applicable.
    max_iter : int, optional
        Maximum number of iterations for the optimizer, defaults to 50.
    run_driver : bool, optional
        If True, the driver will be run, defaults to True.
    make_plots : bool, optional
        If True, generate plots during the optimization, defaults to True.
    phase_info_parameterization : function, optional
        Additional information to parameterize the phase_info object based on
        desired cruise altitude and Mach.

    Returns
    -------
    AviaryProblem
        The AviaryProblem instance after running the optimization problem.

    Notes
    -----
    The function allows for user overrides on aircraft and options data.
    It raises warnings or errors if there are clashing user inputs.
    Users can modify or add methods to alter the Aviary problem's behavior.

    """

    # Build problem
    prob = AviaryProblem(phase_info, mission_method, mass_method, analysis_scheme)

    # Load aircraft and options data from user
    # Allow for user overrides here
    prob.load_inputs(aircraft_filename)

    # Have checks for clashing user inputs
    # Raise warnings or errors depending on how clashing the issues are
    prob.check_inputs()

    prob.add_pre_mission_systems()

    prob.add_phases(phase_info_parameterization=phase_info_parameterization)

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
        record_filename, restart_filename=restart_filename, run_driver=run_driver, make_plots=make_plots, optimization_history_filename=optimization_history_filename)

    return prob


def run_level_1(
    input_deck,
    outdir='output',
    optimizer='SNOPT',
    mass_origin='GASP',
    mission_method='GASP',
    phase_info=None,
    n2=False,
    max_iter=50,
    analysis_scheme=AnalysisScheme.COLLOCATION,
):
    '''
    This file enables running aviary from the command line with a user specified input deck.
    usage: aviary run_mission [input_deck] [opt_args]
    '''

    kwargs = {
        'max_iter': max_iter,
    }

    if analysis_scheme is AnalysisScheme.SHOOTING:
        kwargs['analysis_scheme'] = AnalysisScheme.SHOOTING
        kwargs['run_driver'] = False
    #     kwargs['optimizer'] = 'IPOPT'
    # else:
    kwargs['optimizer'] = optimizer

    if phase_info is None:
        if mission_method == 'GASP':
            from aviary.interface.default_phase_info.gasp import phase_info, phase_info_parameterization
            kwargs['phase_info_parameterization'] = phase_info_parameterization

        else:
            from aviary.interface.default_phase_info.flops import phase_info, phase_info_parameterization
            kwargs['phase_info_parameterization'] = phase_info_parameterization
    else:
        # Load the phase info dynamically from the current working directory
        phase_info_module_path = os.path.join(os.getcwd(), "outputted_phase_info.py")
        if not os.path.exists(phase_info_module_path):
            raise FileNotFoundError(
                "The outputted_phase_info.py file is not in the current working directory. Please run `draw_mission` to generate this file.")

        spec = importlib.util.spec_from_file_location(
            "outputted_phase_info", phase_info_module_path)
        outputted_phase_info = importlib.util.module_from_spec(spec)
        sys.modules["outputted_phase_info"] = outputted_phase_info
        spec.loader.exec_module(outputted_phase_info)

        # Access the phase_info variable from the loaded module
        phase_info = outputted_phase_info.phase_info

    prob = run_aviary(input_deck, phase_info, mission_method=mission_method,
                      mass_method=mass_origin, **kwargs)

    if n2:
        outfile = os.path.join(outdir, "n2.html")
        if outdir != '':
            os.makedirs(outdir, exist_ok=True)
        om.n2(
            prob,
            outfile=outfile,
            show_browser=False,
        )

    return prob


def _setup_level1_parser(parser):
    def_outdir = os.path.join(os.getcwd(), "output")
    parser.add_argument(
        'input_deck', metavar='indeck', type=str, nargs=1, help='Name of vehicle input deck file'
    )
    parser.add_argument(
        "-o", "--outdir", default=def_outdir, help="Directory to write outputs"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default='SNOPT',
        help="Name of optimizer",
        choices=("SNOPT", "IPOPT", "SLSQP", "None")
    )
    parser.add_argument(
        "--mass_origin",
        type=str,
        default="FLOPS",
        help="Mass estimation origin to use",
        choices=("GASP", "FLOPS")
    )
    parser.add_argument(
        "--mission_method",
        type=str,
        default="simple",
        help="Mission origin to use",
        choices=("GASP", "FLOPS", "simple")
    )
    parser.add_argument(
        "--phase_info",
        type=str,
        default=None,
        help="Path to phase info file"
    )
    parser.add_argument("--n2", action="store_true",
                        help="Generate an n2 diagram after the analysis")
    parser.add_argument(
        "--max_iter",
        type=int,
        default=50,
        help="maximum number of iterations")
    parser.add_argument(
        "--shooting",
        action="store_true",
        help="Use shooting instead of collocation",
    )


def _exec_level1(args, user_args):
    if args.shooting:  # For future use
        analysis_scheme = AnalysisScheme.SHOOTING
    else:
        analysis_scheme = AnalysisScheme.COLLOCATION

    if args.optimizer == 'None':
        args.optimizer = None

    # check if args.input_deck is a list, if so, use the first element
    if isinstance(args.input_deck, list):
        args.input_deck = args.input_deck[0]

    prob = run_level_1(
        input_deck=args.input_deck,
        outdir=args.outdir,
        optimizer=args.optimizer,
        mass_origin=args.mass_origin,
        mission_method=args.mission_method,
        phase_info=args.phase_info,
        n2=args.n2,
        max_iter=args.max_iter,
        analysis_scheme=analysis_scheme,
    )
