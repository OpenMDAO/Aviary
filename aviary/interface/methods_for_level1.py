"""This file contains functions needed to run Aviary using the Level 1 interface."""

from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import sys

from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.utils.functions import get_path
from aviary.variable_info.enums import Verbosity


def run_aviary(
    aircraft_data,
    phase_info,
    optimizer=None,
    objective_type=None,
    restart_filename=None,
    max_iter=50,
    run_driver=True,
    make_plots=True,
    phase_info_parameterization=None,
    verbosity=None,
):
    """
    Run the Aviary optimization problem for a specified aircraft configuration and mission.

    This function creates an instance of the AviaryProblem class using provided phase information,
    mission, and mass methods. It processes aircraft and options data from the given aircraft filename,
    checks for clashing inputs, and sets up pre- and post-mission systems. The optimization problem is formulated,
    initialized with guesses, and run to find a solution.

    Parameters
    ----------
    aircraft_data: str, Path, AviaryValues
        Filename from which to load the aircraft and options data, either as a string or
        Path object, or an AviaryValues object containing that information.
    phase_info : dict
        Information about the phases of the mission.
    optimizer : str
        The optimizer to use.
    objective_type : str, optional
        Type of the optimization objective.
    restart_filename : str, optional
        Filename to use for restarting the optimization, if applicable.
    max_iter : int, optional
        Maximum number of iterations for the optimizer, defaults to 50.
    run_driver : bool, optional
        If True, the driver will be run, defaults to True.
    make_plots : bool, optional
        If True, generate plots during the optimization, defaults to True.
    phase_info_parameterization : function, optional
        Additional information to parameterize the phase_info object based on desired cruise
        altitude and Mach.
    verbosity : Verbosity or int, optional
        Sets level of information outputted to the terminal during model execution.
        If provided, overrides verbosity specified in aircraft_data.

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
    # If loading from a file, use filename as problem name. Else, use OpenMDAO default
    if isinstance(aircraft_data, (str, Path)):
        name = Path(aircraft_data).stem
    else:
        name = None

    # Build problem
    prob = AviaryProblem(name=name, verbosity=verbosity)

    # Load aircraft and options data from user
    # Allow for user overrides here
    prob.load_inputs(aircraft_data, phase_info, verbosity=verbosity)

    prob.check_and_preprocess_inputs(verbosity=verbosity)

    # Add Systems
    prob.add_pre_mission_systems(verbosity=verbosity)

    prob.add_phases(phase_info_parameterization=phase_info_parameterization, verbosity=verbosity)

    prob.add_post_mission_systems(verbosity=verbosity)

    # Link phases and variables
    prob.link_phases(verbosity=verbosity)

    prob.add_driver(optimizer, max_iter=max_iter, verbosity=verbosity)

    prob.add_design_variables(verbosity=verbosity)

    # Load optimization problem formulation
    # Detail which variables the optimizer can control
    prob.add_objective(objective_type=objective_type, verbosity=verbosity)

    prob.setup(verbosity=verbosity)

    prob.run_aviary_problem(
        restart_filename=restart_filename,
        run_driver=run_driver,
        make_plots=make_plots,
        verbosity=verbosity,
    )

    return prob


def run_level_1(
    input_deck, optimizer='IPOPT', phase_info=None, max_iter=50, verbosity=Verbosity.BRIEF
):
    """
    This file enables running aviary from the command line with a user specified input deck.
    usage: aviary run_mission [input_deck] [opt_args].
    """
    kwargs = {'max_iter': max_iter, 'optimizer': optimizer, 'verbosity': Verbosity(verbosity)}

    if isinstance(phase_info, str):
        phase_info_path = get_path(phase_info)
        spec = spec_from_file_location('phase_info_file', str(phase_info_path))
        phase_info_file = module_from_spec(spec)
        sys.modules['phase_info_file'] = phase_info_file
        spec.loader.exec_module(phase_info_file)

        phase_info = getattr(phase_info_file, 'phase_info')
        kwargs['phase_info_parameterization'] = getattr(
            phase_info_file, 'phase_info_parameterization', None
        )

    prob = run_aviary(input_deck, phase_info, **kwargs)

    return prob


def _setup_level1_parser(parser):
    parser.add_argument(
        'input_deck',
        metavar='indeck',
        type=str,
        nargs=1,
        help='Name of vehicle input deck file',
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='IPOPT',
        help='Name of optimizer',
        choices=('SNOPT', 'IPOPT', 'SLSQP', 'None'),
    )
    parser.add_argument('--phase_info', type=str, default=None, help='Path to phase info file')
    parser.add_argument('--max_iter', type=int, default=50, help='maximum number of iterations')
    parser.add_argument(
        '--verbosity',
        type=int,
        default=1,
        help='verbosity settings: 0=quiet, 1=brief, 2=verbose, 3=debug',
        choices=(0, 1, 2, 3),
    )


def _exec_level1(args, user_args):
    if args.optimizer == 'None':
        args.optimizer = None

    # check if args.input_deck is a list, if so, use the first element
    if isinstance(args.input_deck, list):
        args.input_deck = args.input_deck[0]

    run_level_1(
        input_deck=args.input_deck,
        optimizer=args.optimizer,
        phase_info=args.phase_info,
        max_iter=args.max_iter,
        verbosity=args.verbosity,
    )
