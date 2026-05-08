import unittest

import numpy as np
from openmdao.core.problem import _clear_problem_names
from openmdao.utils.mpi import MPI
from openmdao.utils.testing_utils import require_pyoptsparse

from aviary.api import Mission
from aviary.interface.run_aviary import run_aviary
from aviary.validation_cases.benchmark_utils import compare_against_expected_values

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

"""This file contains functions needed to run Aviary using the Level 1 interface."""

import os
from pathlib import Path
import subprocess
import sys
from pathlib import Path


def run_aviary(
    aircraft_data,
    phase_info=None,
    optimizer=None,
    objective_type=None,
    subsystems=[],
    restart_filename=None,
    max_iter=50,
    run_driver=True,
    make_plots=True,
    phase_info_modifier=None,
    verbosity=None,
    real_time_plotting=False,
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
        Filename from which to load the aircraft and options data, either as a string or Path
        object, or an AviaryValues object containing that information.
    phase_info : dict, optional
        Information about the phases of the mission.
    optimizer : str, optional
        The optimizer to use.
    objective_type : str, optional
        Type of the optimization objective.
    subsystems : list of SubsystemBuilders, optional
        List of all non-default subsystems to be added to the problem
    restart_filename : str, optional
        Filename to use for restarting the optimization, if applicable.
    max_iter : int, optional
        Maximum number of iterations for the optimizer, defaults to 50.
    run_driver : bool, optional
        If True, the driver will be run, defaults to True.
    make_plots : bool, optional
        If True, generate plots during the optimization, defaults to True.
    phase_info_modifier : function, optional
        Additional information to parameterize the phase_info object, such as based on desired
        cruise altitude and Mach.
    verbosity : Verbosity or int, optional
        Sets level of information outputted to the terminal during model execution. If provided,
        overrides verbosity specified in aircraft_data.

    Returns
    -------
    AviaryProblem
        The AviaryProblem instance after running the optimization problem.

    Notes
    -----
    The function allows for user overrides on aircraft and options data. It raises warnings or
    errors if there are clashing user inputs. Users can modify or add methods to alter the Aviary
    problem's behavior.
    """
    # If loading from a file, use filename as problem name. Else, use OpenMDAO default
    if isinstance(aircraft_data, (str, Path)):
        name = Path(aircraft_data).stem
    else:
        name = None

    from aviary.core.aviary_problem import AviaryProblem

    # Build problem
    prob = AviaryProblem(name=name, verbosity=verbosity)

    # Load aircraft and options data from user
    # Allow for user overrides here
    prob.load_inputs(
        aircraft_data, phase_info, phase_info_modifier=phase_info_modifier, verbosity=verbosity
    )

    prob.load_external_subsystems(subsystems, verbosity=verbosity)

    prob.check_and_preprocess_inputs(verbosity=verbosity)

    # Add Systems
    prob.add_pre_mission_systems(verbosity=verbosity)

    prob.add_phases(verbosity=verbosity)

    prob.add_post_mission_systems(verbosity=verbosity)

    # Link phases and variables
    prob.link_phases(verbosity=verbosity)

    prob.add_driver(optimizer, max_iter=max_iter, verbosity=verbosity)

    prob.add_design_variables(verbosity=verbosity)

    ## Load optimization problem formulation
    ## Detail which variables the optimizer can control
    prob.add_objective(objective_type=objective_type, verbosity=verbosity)

    prob.setup(verbosity=verbosity)
    prob.final_setup()

    prob.run_aviary_problem(
        restart_filename=restart_filename,
        run_driver=run_driver,
        make_plots=make_plots,
        verbosity=verbosity,
        real_time_plotting=real_time_plotting,
    )

    return prob




class ProblemPhaseTestCase(unittest.TestCase):
    """
    Setup of a large single aisle commercial transport aircraft using
    FLOPS mass and aero method and ENERGY_STATE mission method. Expected outputs based
    on 'validation_cases/validation_data/test_models/aircraft_for_bench_FwFm.csv' model.
    """

    def setUp(self):
        expected_dict = {}

        # block auto-formatting of tables
        # fmt: off
        expected_dict['times'] = np.array(
            [
                [120.],[163.76268404], [224.14625594], [243.25744998], [243.25744998],
                [336.40804126], [464.93684491], [505.61577182], [505.61577182],  [626.46953842],
                [793.22306972], [845.99999224], [845.99999224], [966.85375884], [1133.60729014],
                [1186.38421266], [1186.38421266], [1279.53480393], [1408.06360758], [1448.74253449],
                [1448.74253449], [1492.50521853], [1552.88879042], [1571.99998447], [1571.99998447],
                [10224.87383109], [22164.07366288], [25942.78958866], [25942.78958866],
                [26009.11685074], [26100.63493484], [26129.60009555], [26129.60009555],
                [26265.05921709], [26451.96515722], [26511.12024823], [26511.12024823],
                [26672.16774132], [26894.38041154], [26964.7099619], [26964.7099619],
                [27100.16908344], [27287.07502357], [27346.23011458], [27346.23011458],
                [27412.55737667], [27504.07546076], [27533.04062147]
            ]
        )

        expected_dict['altitudes'] = np.array(
            [
                [10.668], [0.], [1001.70617719], [1429.27176545], [1429.27176545], [3413.27102762],
                [5642.3831233], [6169.75300447], [6169.75300447], [7399.140983], [8514.78661356],
                [8803.21405264], [8803.21405264], [9373.68426297], [10020.99237958],
                [10196.42552457], [10196.42552457],[10451.72258036], [10652.38789684], [10668.],
                [10668.], [10660.42246376], [10656.16585151], [10668.], [10668.], [10668.],
                [10668.], [10668.], [10668.], [10668.], [10142.11478951], [9922.15743555],
                [9922.15743555], [8891.66886638], [7502.1861348], [7069.1900852], [7069.1900852],
                [5896.44637998], [4264.29354306], [3737.8471594], [3737.8471594], [2702.15624637],
                [1248.18960736], [793.03526817], [793.03526817], [345.06939295], [10.668], [10.668]
            ]
        )

        expected_dict['masses'] = np.array(
            [
                [79303.30184763], [79221.39668215], [79075.19453181], [79028.6003426],
                [79028.6003426], [78828.82221909], [78613.60466821], [78557.84739563],
                [78557.84739563], [78411.06578989], [78238.0916773], [78186.75440341],
                [78186.75440341], [78077.23953313], [77938.37965175], [77896.59718975],
                [77896.59718975], [77825.81832958], [77732.75016916], [77704.11629998],
                [77704.11629998], [77673.32196072], [77630.75735319], [77617.25716885],
                [77617.25716885], [72178.78521803], [65072.41395049], [62903.84179505],
                [62903.84179505], [62896.27636813], [62888.3612195], [62885.93748938],
                [62885.93748938], [62874.48788511], [62857.70600096], [62852.13740881],
                [62852.13740881], [62835.97069937], [62810.37776063], [62801.1924259],
                [62801.1924259], [62781.32471014], [62748.91017128], [62737.32520462],
                [62737.32520462], [62723.59895849], [62703.94977811], [62697.71513264]
            ]
        )

        expected_dict['ranges'] = np.array(
            [
                [1452.84514351], [6093.51223933], [15820.03029119], [19123.61258676],
                [19123.61258676], [36374.65336952], [61265.3984918], [69106.49687132],
                [69106.49687132], [92828.04820577], [126824.13801408], [138011.02420534],
                [138011.02420534], [164027.18014424], [200524.66550565], [212113.49107256],
                [212113.49107256], [232622.50720766], [261189.53466522], [270353.40501262],
                [270353.40501262], [280350.48472685], [294356.27080588], [298832.61221641],
                [298832.61221641], [2325837.11255987], [5122689.60556392], [6007883.85695889],
                [6007883.85695889], [6022237.43153219], [6039575.06318219], [6044873.89820027],
                [6044873.89820027], [6068553.1921364], [6099290.23732297], [6108673.67260778],
                [6108673.67260778],  [6133535.09572671], [6166722.19545137], [6177077.72115854],
                [6177077.72115854], [6197011.1330154], [6224357.63792683], [6232920.45309764],
                [6232920.45309764], [6242332.46480721], [6254144.50957549], [6257352.4]
            ]
        )

        expected_dict['velocities'] = np.array(
            [
                [69.30879167], [137.49019035], [174.54683946], [179.28863383], [179.28863383],
                [191.76748742], [194.33322917], [194.52960387], [194.52960387], [199.01184603],
                [209.81696863], [212.86546124], [212.86546124], [217.37467051], [219.67762167],
                [219.97194272], [219.97194272], [220.67963782], [224.38113484], [226.77184704],
                [226.77184704], [230.01128033], [233.72454583], [234.25795132], [234.25795132],
                [234.25795132], [234.25795132], [234.25795132], [234.25795132], [201.23881],
                [182.84158341], [180.10650108], [180.10650108], [169.77497514], [159.59034446],
                [157.09907013], [157.09907013], [151.659491], [147.52098882], [147.07683999],
                [147.07683999], [147.05392009], [145.31556891], [143.47446173], [143.47446173],
                [138.99109332], [116.22447082], [102.07377559]
            ]
        )
        # fmt: on

        self.expected_dict = expected_dict

        phase_info = {
            'pre_mission': {'include_takeoff': True, 'optimize_mass': True},
            'climb': {
                'subsystem_options': {'aerodynamics': {'method': 'computed'}},
                'user_options': {
                    'num_segments': 6,
                    'order': 3,
                    'mach_bounds': ((0.1, 0.8), 'unitless'),
                    'mach_optimize': True,
                    'altitude_bounds': ((0.0, 35000.0), 'ft'),
                    'altitude_optimize': True,
                    'throttle_enforcement': 'path_constraint',
                    'mass_ref': (200000, 'lbm'),
                    'time_initial': (0.0, 'min'),
                    'time_duration_bounds': ((20.0, 60.0), 'min'),
                    'no_descent': True,
                },
                'initial_guesses': {
                    'time': ([0, 40.0], 'min'),
                    'altitude': ([35, 35000.0], 'ft'),
                    'mach': ([0.3, 0.79], 'unitless'),
                },
            },
            'cruise': {
                'subsystem_options': {'aerodynamics': {'method': 'computed'}},
                'user_options': {
                    'num_segments': 1,
                    'order': 3,
                    'mach_initial': (0.79, 'unitless'),
                    'mach_bounds': ((0.79, 0.79), 'unitless'),
                    'mach_optimize': True,
                    'mach_polynomial_order': 1,
                    'altitude_initial': (35000.0, 'ft'),
                    'altitude_bounds': ((35000.0, 35000.0), 'ft'),
                    'altitude_optimize': True,
                    'altitude_polynomial_order': 1,
                    'throttle_enforcement': 'boundary_constraint',
                    'mass_ref': (200000, 'lbm'),
                    'time_initial_bounds': ((20.0, 60.0), 'min'),
                    'time_duration_bounds': ((60.0, 720.0), 'min'),
                },
                'initial_guesses': {
                    'time': ([40, 200], 'min'),
                    'altitude': ([35000, 35000.0], 'ft'),
                    'mach': ([0.79, 0.79], 'unitless'),
                },
            },
            'descent': {
                'subsystem_options': {'aerodynamics': {'method': 'computed'}},
                'user_options': {
                    'num_segments': 5,
                    'order': 3,
                    'mach_initial': (0.79, 'unitless'),
                    'mach_final': (0.3, 'unitless'),
                    'mach_bounds': ((0.2, 0.8), 'unitless'),
                    'mach_optimize': True,
                    'altitude_initial': (35000.0, 'ft'),
                    'altitude_final': (35.0, 'ft'),
                    'altitude_bounds': ((0.0, 35000.0), 'ft'),
                    'altitude_optimize': True,
                    'throttle_enforcement': 'path_constraint',
                    'mass_ref': (200000, 'lbm'),
                    'distance_ref': (3375, 'nmi'),
                    'time_initial_bounds': ((80.0, 780.0), 'min'),
                    'time_duration_bounds': ((5.0, 45.0), 'min'),
                    'no_climb': True,
                },
                'initial_guesses': {
                    'time': ([240, 30], 'min'),
                },
            },
            'post_mission': {
                'include_landing': True,
                'constrain_range': True,
                'target_range': (3375.0, 'nmi'),
            },
        }

        self.phase_info = phase_info

        _clear_problem_names()  # need to reset these to simulate separate runs


class TestBenchFwFmParallel(ProblemPhaseTestCase):
    """Run the model in parallel that is setup in ProblemPhaseTestCase class."""

    N_PROCS = 3

    @require_pyoptsparse(optimizer='IPOPT')
    def test_bench_FwFm_SNOPT_MPI(self):

        prob = run_aviary(
            'validation_cases/validation_data/test_models/aircraft_for_bench_FwFm.csv',
            self.phase_info,
            verbosity=2,
            max_iter=1,
            optimizer='IPOPT',
        )
        # self.assertTrue(prob.result.success)
        #compare_against_expected_values(prob, self.expected_dict)


if __name__ == '__main__':
    # unittest.main()
    test = TestBenchFwFmSerial()
    test.setUp()
    # test.test_bench_FwFm_IPOPT()
    test.test_bench_FwFm_SNOPT()
