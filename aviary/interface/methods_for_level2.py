import csv
import inspect
import json
import os
import warnings
from datetime import datetime
from enum import Enum
from pathlib import Path

import dymos as dm
import numpy as np
import openmdao.api as om
from openmdao.utils.reports_system import _default_reports

from aviary.core.aviary_group import AviaryGroup

from aviary.utils.aviary_values import AviaryValues
from aviary.utils.functions import convert_strings_to_data
from aviary.interface.utils import set_warning_format
from aviary.utils.merge_variable_metadata import merge_meta_data

from aviary.variable_info.enums import (
    EquationsOfMotion,
    LegacyCode,
    ProblemType,
    Verbosity,
)
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variable_meta_data import _MetaData as BaseMetaData
from aviary.variable_info.variables import Aircraft, Dynamic, Mission, Settings

FLOPS = LegacyCode.FLOPS
GASP = LegacyCode.GASP


class AviaryProblem(om.Problem):
    """
    Main class for instantiating, formulating, and solving Aviary problems.

    On a basic level, this problem object is all the conventional user needs
    to interact with. Looking at the three "levels" of use cases, from simplest
    to most complicated, we have:

    Level 1: users interact with Aviary through input files (.csv or .yaml, TBD)
    Level 2: users interact with Aviary through a Python interface
    Level 3: users can modify Aviary's workings through Python and OpenMDAO

    This Problem object is simply a specialized OpenMDAO Problem that has
    additional methods to help users create and solve Aviary problems.
    """

    def __init__(self, verbosity=None, **kwargs):
        # Modify OpenMDAO's default_reports for this session.
        new_reports = [
            'subsystems',
            'mission',
            'timeseries_csv',
            'run_status',
            'input_checks',
        ]
        for report in new_reports:
            if report not in _default_reports:
                _default_reports.append(report)

        super().__init__(**kwargs)

        self.timestamp = datetime.now()

        # If verbosity is set to anything but None, this defines how warnings are formatted for the
        # whole problem - warning format won't be updated if user requests a different verbosity
        # level for a specific method
        self.verbosity = verbosity
        set_warning_format(verbosity)

        self.model = AviaryGroup()

        self.aviary_inputs = None

    def load_inputs(
        self,
        aircraft_data,
        phase_info=None,
        engine_builders=None,
        problem_configurator=None,
        meta_data=BaseMetaData,
        verbosity=None,
    ):
        """
        This method loads the aviary_values inputs and options that the user specifies. They could
        specify files to load and values to replace here as well.

        Phase info is also loaded if provided by the user. If phase_info is None, the appropriate
        default phase_info based on mission analysis method is used.

        This method is not strictly necessary; a user could also supply an AviaryValues object
        and/or phase_info dict of their own.
        """
        # We haven't read the input data yet, we don't know what desired run verbosity is
        # `self.verbosity` is "true" verbosity for entire run. `verbosity` is verbosity
        # override for just this method
        if verbosity is not None:
            # compatibility with being passed int for verbosity
            verbosity = Verbosity(verbosity)
        else:
            verbosity = self.verbosity  # usually None

        # TODO: We cannot pass self.verbosity back up from load inputs for multi-mission because there could be multiple .csv files
        aviary_inputs, verbosity = self.model.load_inputs(
            aircraft_data=aircraft_data,
            phase_info=phase_info,
            engine_builders=engine_builders,
            problem_configurator=problem_configurator,
            meta_data=meta_data,
            verbosity=verbosity,
        )

        # When there is only 1 aircraft model/mission, preserve old behavior.
        self.phase_info = self.model.phase_info
        self.aviary_inputs = aviary_inputs
        self.verbosity = verbosity

        return self.aviary_inputs

    def check_and_preprocess_inputs(self, verbosity=None):
        """
        This method checks the user-supplied input values for any potential problems
        and preprocesses the inputs to prepare them for use in the Aviary problem.
        """

        # `self.verbosity` is "true" verbosity for entire run. `verbosity` is verbosity
        # override for just this method
        if verbosity is not None:
            # compatibility with being passed int for verbosity
            verbosity = Verbosity(verbosity)
        else:
            verbosity = self.verbosity  # defaults to BRIEF

        self.model.check_and_preprocess_inputs(verbosity=verbosity)

        self._update_metadata_from_subsystems()

    def _update_metadata_from_subsystems(self):
        """Merge metadata from user-defined subsystems into problem metadata."""
        self.meta_data = BaseMetaData.copy()

        # loop through phase_info and external subsystems
        for phase_name in self.model.phase_info:
            # TODO: phase_info now resides in AviaryGroup. Accessing it as self.model.phase_info is just a temporary stop-gap
            # it will be necessary to combine multiple self.models
            external_subsystems = self.model.get_all_subsystems(
                self.model.phase_info[phase_name]['external_subsystems']
            )

            for subsystem in external_subsystems:
                meta_data = subsystem.meta_data.copy()
                self.meta_data = merge_meta_data([self.meta_data, meta_data])

        self.model.meta_data = self.meta_data  # TODO: temporary fix

    def add_pre_mission_systems(self, verbosity=None):
        """
        Add pre-mission systems to the Aviary problem. These systems are executed before
        the mission.

        Depending on the mission model specified (`FLOPS` or `GASP`), this method adds
        various subsystems to the aircraft model. For the `FLOPS` mission model, a
        takeoff phase is added using the Takeoff class with the number of engines and
        airport altitude specified. For the `GASP` mission model, three subsystems are
        added: a TaxiSegment subsystem, an ExecComp to calculate the time to initiate
        gear and flaps, and an ExecComp to calculate the speed at which to initiate
        rotation. All subsystems are promoted with aircraft and mission inputs and
        outputs as appropriate.

        A user can override this method with their own pre-mission systems as desired.
        """
        # `self.verbosity` is "true" verbosity for entire run. `verbosity` is verbosity
        # override for just this method
        if verbosity is not None:
            # compatibility with being passed int for verbosity
            verbosity = Verbosity(verbosity)
        else:
            verbosity = self.verbosity  # defaults to BRIEF

        self.model.add_pre_mission_systems(verbosity=verbosity)

    def add_phases(
        self,
        phase_info_parameterization=None,
        parallel_phases=True,
        verbosity=None,
    ):
        """
        Add the mission phases to the problem trajectory based on the user-specified
        phase_info dictionary.

        Parameters
        ----------
        phase_info_parameterization (function, optional): A function that takes in the
            phase_info dictionary and aviary_inputs and returns modified phase_info.
            Defaults to None.

        parallel_phases (bool, optional): If True, the top-level container of all phases
            will be a ParallelGroup, otherwise it will be a standard OpenMDAO Group.
            Defaults to True.

        Returns
        -------
        <Trajectory>
            The Dymos Trajectory object containing the added mission phases.
        """
        # `self.verbosity` is "true" verbosity for entire run. `verbosity` is verbosity
        # override for just this method
        if verbosity is not None:
            # compatibility with being passed int for verbosity
            verbosity = Verbosity(verbosity)
        else:
            verbosity = self.verbosity  # defaults to BRIEF

        return self.model.add_phases(
            phase_info_parameterization=phase_info_parameterization,
            parallel_phases=parallel_phases,
            verbosity=verbosity,
            comm=self.comm,
        )

    def add_post_mission_systems(self, verbosity=None):
        """
        Add post-mission systems to the aircraft model. This is akin to the pre-mission
        group or the "premission_systems", but occurs after the mission in the execution
        order.

        Depending on the mission model specified (`FLOPS` or `GASP`), this method adds
        various subsystems to the aircraft model. For the `FLOPS` mission model, a
        landing phase is added using the Landing class with the wing area and lift
        coefficient specified, and a takeoff constraints ExecComp is added to enforce
        mass, range, velocity, and altitude continuity between the takeoff and climb
        phases. The landing subsystem is promoted with aircraft and mission inputs and
        outputs as appropriate, while the takeoff constraints ExecComp is only promoted
        with mission inputs and outputs.

        For the `GASP` mission model, four subsystems are added: a LandingSegment
        subsystem, an ExecComp to calculate the reserve fuel required, an ExecComp to
        calculate the overall fuel burn, and three ExecComps to calculate various
        mission objectives and constraints. All subsystems are promoted with aircraft
        and mission inputs and outputs as appropriate.

        A user can override this with their own postmission systems.
        """
        # `self.verbosity` is "true" verbosity for entire run. `verbosity` is verbosity
        # override for just this method
        if verbosity is not None:
            # compatibility with being passed int for verbosity
            verbosity = Verbosity(verbosity)
        else:
            verbosity = self.verbosity  # defaults to BRIEF

        self.model.add_post_mission_systems(verbosity=verbosity)

    def link_phases(self, verbosity=None):
        """
        Link phases together after they've been added.

        Based on which phases the user has selected, we might need
        special logic to do the Dymos linkages correctly. Some of those
        connections for the simple GASP and FLOPS mission are shown here.
        """
        # `self.verbosity` is "true" verbosity for entire run. `verbosity` is verbosity
        # override for just this method
        if verbosity is not None:
            # compatibility with being passed int for verbosity
            verbosity = Verbosity(verbosity)
        else:
            verbosity = self.verbosity  # defaults to BRIEF

        self.model.link_phases(verbosity=verbosity, comm=self.comm)

    def add_driver(self, optimizer=None, use_coloring=None, max_iter=50, verbosity=None):
        """
        Add an optimization driver to the Aviary problem.

        Depending on the provided optimizer, the method instantiates the relevant
        driver (ScipyOptimizeDriver or pyOptSparseDriver) and sets the optimizer
        options. Options for 'SNOPT', 'IPOPT', and 'SLSQP' are specified. The method
        also allows for the declaration of coloring and setting debug print options.

        Parameters
        ----------
        optimizer : str
            The name of the optimizer to use. It can be "SLSQP", "SNOPT", "IPOPT" or
            others supported by OpenMDAO. If "SLSQP", it will instantiate a
            ScipyOptimizeDriver, else it will instantiate a pyOptSparseDriver.

        use_coloring : bool, optional
            If True (default), the driver will declare coloring, which can speed up
            derivative computations.

        max_iter : int, optional
            The maximum number of iterations allowed for the optimization process.
            Default is 50. This option is applicable to "SNOPT", "IPOPT", and "SLSQP"
            optimizers.

        verbosity : Verbosity or int, optional
            Controls the level of printouts for this method. If None, uses the value of
            Settings.VERBOSITY in provided aircraft data.

        Returns
        -------
        None
        """
        # `self.verbosity` is "true" verbosity for entire run. `verbosity` is verbosity
        # override for just this method
        if verbosity is not None:
            # compatibility with being passed int for verbosity
            verbosity = Verbosity(verbosity)
        else:
            verbosity = self.verbosity  # defaults to BRIEF

        # Set defaults for optimizer and use_coloring
        if optimizer is None:
            optimizer = 'IPOPT'
        if use_coloring is None:
            use_coloring = True

        # check if optimizer is SLSQP
        if optimizer == 'SLSQP':
            driver = self.driver = om.ScipyOptimizeDriver()
        else:
            driver = self.driver = om.pyOptSparseDriver()

        driver.options['optimizer'] = optimizer
        if use_coloring:
            # define coloring options by verbosity
            if verbosity < Verbosity.VERBOSE:  # QUIET, BRIEF
                driver.declare_coloring(show_summary=False)
            elif verbosity == Verbosity.VERBOSE:
                driver.declare_coloring(show_summary=True)
            else:  # DEBUG
                driver.declare_coloring(show_summary=True, show_sparsity=True)

        if driver.options['optimizer'] == 'SNOPT':
            # Print Options #
            if verbosity == Verbosity.QUIET:
                isumm, iprint = 0, 0
            elif verbosity == Verbosity.BRIEF:
                isumm, iprint = 6, 0
            elif verbosity > Verbosity.BRIEF:  # VERBOSE, DEBUG
                isumm, iprint = 6, 9
            driver.opt_settings['iSumm'] = isumm
            driver.opt_settings['iPrint'] = iprint
            # Optimizer Settings #
            driver.opt_settings['Major iterations limit'] = max_iter
            driver.opt_settings['Major optimality tolerance'] = 1e-4
            driver.opt_settings['Major feasibility tolerance'] = 1e-7

        elif driver.options['optimizer'] == 'IPOPT':
            # Print Options #
            if verbosity == Verbosity.QUIET:
                print_level = 0
                driver.opt_settings['print_user_options'] = 'no'
            elif verbosity == Verbosity.BRIEF:
                print_level = 3  # minimum to get exit status
                driver.opt_settings['print_user_options'] = 'no'
                driver.opt_settings['print_frequency_iter'] = 10
            elif verbosity == Verbosity.VERBOSE:
                print_level = 5
            else:  # DEBUG
                print_level = 7
            driver.opt_settings['print_level'] = print_level
            # Optimizer Settings #
            driver.opt_settings['tol'] = 1.0e-6
            driver.opt_settings['mu_init'] = 1e-5
            driver.opt_settings['max_iter'] = max_iter
            # for faster convergence
            driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
            driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'
            driver.opt_settings['mu_strategy'] = 'monotone'

        elif driver.options['optimizer'] == 'SLSQP':
            # Print Options #
            if verbosity == Verbosity.QUIET:
                disp = False
            else:
                disp = True
            driver.options['disp'] = disp
            # Optimizer Settings #
            driver.options['tol'] = 1e-9
            driver.options['maxiter'] = max_iter

        # pyoptsparse print settings for both SNOPT, IPOPT
        if optimizer in ('SNOPT', 'IPOPT'):
            if verbosity == Verbosity.QUIET:
                driver.options['print_results'] = False
            elif verbosity < Verbosity.DEBUG:  # QUIET, BRIEF, VERBOSE
                driver.options['print_results'] = 'minimal'
            elif verbosity >= Verbosity.DEBUG:
                driver.options['print_opt_prob'] = True

        # optimizer agnostic settings
        if verbosity >= Verbosity.VERBOSE:  # VERBOSE, DEBUG
            driver.options['debug_print'] = ['desvars']
            if verbosity == Verbosity.DEBUG:
                driver.options['debug_print'] = [
                    'desvars',
                    'ln_cons',
                    'nl_cons',
                    'objs',
                ]

    def add_design_variables(self, verbosity=None):
        """
        Adds design variables to the Aviary problem.

        Depending on the mission model and problem type, different design variables and
        constraints are added.

        If using the FLOPS model, a design variable is added for the gross mass of the
        aircraft, with a lower bound of 10 lbm and an upper bound of 900,000 lbm.

        If using the GASP model, the following design variables are added depending on
        the mission type:
        - the initial thrust-to-weight ratio of the aircraft during ascent
        - the duration of the ascent phase
        - the time constant for the landing gear actuation
        - the time constant for the flaps actuation

        In addition, two constraints are added for the GASP model:
        - the initial altitude of the aircraft with gear extended is constrained to be 50 ft
        - the initial altitude of the aircraft with flaps extended is constrained to be 400 ft

        If solving a sizing problem, a design variable is added for the gross mass of
        the aircraft, and another for the gross mass of the aircraft computed during the
        mission. A constraint is also added to ensure that the residual range is zero.

        If solving an alternate problem, only a design variable for the gross mass of
        the aircraft computed during the mission is added. A constraint is also added to
        ensure that the residual range is zero.

        In all cases, a design variable is added for the final cruise mass of the
        aircraft, with no upper bound, and a residual mass constraint is added to ensure
        that the mass balances.

        """
        # `self.verbosity` is "true" verbosity for entire run. `verbosity` is verbosity
        # override for just this method
        if verbosity is not None:
            # compatibility with being passed int for verbosity
            verbosity = Verbosity(verbosity)
        else:
            verbosity = self.verbosity  # defaults to BRIEF

        self.model.add_design_variables(verbosity=verbosity)

    def add_objective(self, objective_type=None, ref=None, verbosity=None):
        """
        Add the objective function based on the given objective_type and ref.

        NOTE: the ref value should be positive for values you're trying
        to minimize and negative for values you're trying to maximize.
        Please check and double-check that your ref value makes sense
        for the objective you're using.

        Parameters
        ----------
        objective_type : str
            The type of objective to add. Options are 'mass', 'hybrid_objective',
            'fuel_burned', and 'fuel'.
        ref : float
            The reference value for the objective. If None, a default value will be used
            based on the objective type. Please see the `default_ref_values` dict for
            these default values.
        verbosity : Verbosity or int, optional
            Controls the level of printouts for this method. If None, uses the value of
            Settings.VERBOSITY in provided aircraft data.

        Raises
        ------
            ValueError: If an invalid problem type is provided.

        """
        # `self.verbosity` is "true" verbosity for entire run. `verbosity` is verbosity
        # override for just this method
        if verbosity is not None:
            # compatibility with being passed int for verbosity
            verbosity = Verbosity(verbosity)
        else:
            verbosity = self.verbosity  # defaults to BRIEF

        self.model.add_subsystem(
            'fuel_obj',
            om.ExecComp(
                'reg_objective = overall_fuel/10000 + ascent_duration/30.',
                reg_objective={'val': 0.0, 'units': 'unitless'},
                ascent_duration={'units': 's', 'shape': 1},
                overall_fuel={'units': 'lbm'},
            ),
            promotes_inputs=[
                ('ascent_duration', Mission.Takeoff.ASCENT_DURATION),
                ('overall_fuel', Mission.Summary.TOTAL_FUEL_MASS),
            ],
            promotes_outputs=[('reg_objective', Mission.Objectives.FUEL)],
        )

        # TODO: All references to self.model. will need to be updated
        self.model.add_subsystem(
            'range_obj',
            om.ExecComp(
                'reg_objective = -actual_range/1000 + ascent_duration/30.',
                reg_objective={'val': 0.0, 'units': 'unitless'},
                ascent_duration={'units': 's', 'shape': 1},
                actual_range={'val': self.model.target_range, 'units': 'NM'},
            ),
            promotes_inputs=[
                ('actual_range', Mission.Summary.RANGE),
                ('ascent_duration', Mission.Takeoff.ASCENT_DURATION),
            ],
            promotes_outputs=[('reg_objective', Mission.Objectives.RANGE)],
        )

        # Dictionary for default reference values
        default_ref_values = {
            'mass': -5e4,
            'hybrid_objective': -5e4,
            'fuel_burned': 1e4,
            'fuel': 1e4,
        }

        # Check if an objective type is specified
        if objective_type is not None:
            ref = ref if ref is not None else default_ref_values.get(objective_type, 1)

            final_phase_name = self.model.regular_phases[-1]

            if objective_type == 'mass':
                self.model.add_objective(
                    f'traj.{final_phase_name}.timeseries.{Dynamic.Vehicle.MASS}',
                    index=-1,
                    ref=ref,
                )
            elif objective_type == 'time':
                self.model.add_objective(
                    f'traj.{final_phase_name}.timeseries.time', index=-1, ref=ref
                )

            elif objective_type == 'hybrid_objective':
                self._add_hybrid_objective(self.model.phase_info)
                self.model.add_objective('obj_comp.obj')

            elif objective_type == 'fuel_burned':
                self.model.add_objective(Mission.Summary.FUEL_BURNED, ref=ref)

            elif objective_type == 'fuel':
                self.model.add_objective(Mission.Objectives.FUEL, ref=ref)

            else:
                raise ValueError(
                    f"{objective_type} is not a valid objective. 'objective_type' must "
                    'be one of the following: mass, time, hybrid_objective, '
                    'fuel_burned, or fuel'
                )

        else:  # If no 'objective_type' is specified, we handle based on 'problem_type'
            # If 'ref' is not specified, assign a default value
            ref = ref if ref is not None else 1

            if self.model.problem_type is ProblemType.SIZING:
                self.model.add_objective(Mission.Objectives.FUEL, ref=ref)

            elif self.model.problem_type is ProblemType.ALTERNATE:
                self.model.add_objective(Mission.Objectives.FUEL, ref=ref)

            elif self.model.problem_type is ProblemType.FALLOUT:
                self.model.add_objective(Mission.Objectives.RANGE, ref=ref)

            else:
                raise ValueError(f'{self.model.problem_type} is not a valid problem type.')

    def setup(self, **kwargs):
        """Lightly wrapped setup() method for the problem."""
        # verbosity is not used in this method, but it is understandable that a user
        # might try and include it (only method that doesn't accept it). Capture it
        if 'verbosity' in kwargs:
            kwargs.pop('verbosity')
        # Use OpenMDAO's model options to pass all options through the system hierarchy.
        setup_model_options(self, self.aviary_inputs, self.meta_data)

        # suppress warnings:
        # "input variable '...' promoted using '*' was already promoted using 'aircraft:*'
        # TODO: will need to setup warnings on each AviaryGroup()
        with warnings.catch_warnings():
            self.model.options['aviary_options'] = self.aviary_inputs
            self.model.options['aviary_metadata'] = self.meta_data
            self.model.options['phase_info'] = self.model.phase_info

            warnings.simplefilter('ignore', om.OpenMDAOWarning)
            warnings.simplefilter('ignore', om.PromotionWarning)

            super().setup(**kwargs)

    def set_initial_guesses(self, parent_prob=None, parent_prefix='', verbosity=None):
        """
        Call `set_val` on the trajectory for states and controls to seed the problem with
        reasonable initial guesses. This is especially important for collocation methods.

        This method first identifies all phases in the trajectory then loops over each phase.
        Specific initial guesses are added depending on the phase and mission method. Cruise is
        treated as a special phase for GASP-based missions because it is an AnalyticPhase in
        Dymos. For this phase, we handle the initial guesses first separately and continue to the
        next phase after that. For other phases, we set the initial guesses for states and
        controls according to the information available in the 'initial_guesses' attribute of the
        phase.
        """
        # `self.verbosity` is "true" verbosity for entire run. `verbosity` is verbosity
        # override for just this method
        if verbosity is not None:
            # compatibility with being passed int for verbosity
            verbosity = Verbosity(verbosity)
        else:
            verbosity = self.verbosity  # defaults to BRIEF

        self.model.set_initial_guesses(
            parent_prob=parent_prob,
            parent_prefix=parent_prefix,
            verbosity=verbosity,
        )

    def run_aviary_problem(
        self,
        record_filename='problem_history.db',
        optimization_history_filename=None,
        restart_filename=None,
        suppress_solver_print=True,
        run_driver=True,
        simulate=False,
        make_plots=True,
        verbosity=None,
    ):
        """
        This function actually runs the Aviary problem, which could be a simulation,
        optimization, or a driver execution, depending on the arguments provided.

        Parameters
        ----------
        record_filename : str, optional
            The name of the database file where the solutions are to be recorded. The
            default is "problem_history.db".
        optimization_history_filename : str, None
            The name of the database file where the driver iterations are to be
            recorded. The default is None.
        restart_filename : str, optional
            The name of the file that contains previously computed solutions which are
            to be used as starting points for this run. If it is None (default), no
            restart file will be used.
        suppress_solver_print : bool, optional
            If True (default), all solvers' print statements will be suppressed. Useful
            for deeply nested models with multiple solvers so the print statements don't
            overwhelm the output.
        run_driver : bool, optional
            If True (default), the driver (aka optimizer) will be executed. If False,
            the problem will be run through one pass -- equivalent to OpenMDAO's
            `run_model` behavior.
        simulate : bool, optional
            If True, an explicit Dymos simulation will be performed. The default is
            False.
        make_plots : bool, optional
            If True (default), Dymos html plots will be generated as part of the output.
        """
        # `self.verbosity` is "true" verbosity for entire run. `verbosity` is verbosity
        # override for just this method
        if verbosity is not None:
            # compatibility with being passed int for verbosity
            verbosity = Verbosity(verbosity)
        else:
            verbosity = self.verbosity  # defaults to BRIEF

        if verbosity >= Verbosity.VERBOSE:  # VERBOSE, DEBUG
            self.final_setup()
            with open('input_list.txt', 'w') as outfile:
                self.model.list_inputs(out_stream=outfile)

        if suppress_solver_print:
            self.set_solver_print(level=0)

        if optimization_history_filename:
            recorder = om.SqliteRecorder(optimization_history_filename)
            self.driver.add_recorder(recorder)

        # and run mission, and dynamics
        if run_driver:
            failed = dm.run_problem(
                self,
                run_driver=run_driver,
                simulate=simulate,
                make_plots=make_plots,
                solution_record_file=record_filename,
                restart=restart_filename,
            )

            # TODO this is only used in a single test. Either self.problem_ran_successfully
            #      should be removed, or rework this option to be more helpful (store
            # entire "failed" object?) and implement more rigorously in benchmark
            # tests
            if failed.exit_status == 'FAIL':
                self.problem_ran_successfully = False
            else:
                self.problem_ran_successfully = True
            # Manually print out a failure message for low verbosity modes that suppress
            # optimizer printouts, which may include the results message. Assumes success,
            # alerts user on a failure
            if (
                not self.problem_ran_successfully and verbosity <= Verbosity.BRIEF  # QUIET, BRIEF
            ):
                warnings.warn('\nAviary run failed. See the dashboard for more details.\n')
        else:
            # prevent UserWarning that is displayed when an event is triggered
            warnings.filterwarnings('ignore', category=UserWarning)
            # TODO failed doesn't exist for run_model(), no return from method
            failed = self.run_model()
            warnings.filterwarnings('default', category=UserWarning)

        # update n2 diagram after run.
        outdir = Path(self.get_reports_dir(force=True))
        outfile = os.path.join(outdir, 'n2.html')
        om.n2(
            self,
            outfile=outfile,
            show_browser=False,
        )

        if verbosity >= Verbosity.VERBOSE:  # VERBOSE, DEBUG
            with open('output_list.txt', 'w') as outfile:
                self.model.list_outputs(out_stream=outfile)

        self.problem_ran_successfully = not failed

    def alternate_mission(
        self,
        run_mission=True,
        json_filename='sizing_problem.json',
        num_first=None,
        num_business=None,
        num_tourist=None,
        num_pax=None,
        wing_cargo=None,
        misc_cargo=None,
        cargo_mass=None,
        mission_range=None,
        phase_info=None,
        verbosity=None,
    ):
        """
        This function runs an alternate mission based on a sizing mission output.

        Parameters
        ----------
        run_mission : bool
            Flag to determine whether to run the mission before returning the problem
            object.
        json_filename : str
            Name of the file that the sizing mission has been saved to.
        mission_range : float, optional
            Target range for the fallout mission.
        payload_mass : float, optional
            Mass of the payload for the mission.
        phase_info : dict, optional
            Dictionary containing the phases and their required parameters.
        verbosity : Verbosity or int, optional
            Controls the level of printouts for this method. If None, uses the value of
            Settings.VERBOSITY in provided aircraft data.
        """
        # `self.verbosity` is "true" verbosity for entire run. `verbosity` is verbosity
        # override for just this method
        if verbosity is not None:
            # compatibility with being passed int for verbosity
            verbosity = Verbosity(verbosity)
        else:
            verbosity = self.verbosity  # defaults to BRIEF

        # TODO: these self.aviary_inputs methods will need to be updated
        mass_method = self.aviary_inputs.get_val(Settings.MASS_METHOD)
        equations_of_motion = self.aviary_inputs.get_val(Settings.EQUATIONS_OF_MOTION)
        if mass_method == LegacyCode.FLOPS:
            if num_first is None or num_business is None or num_tourist is None:
                if verbosity > Verbosity.BRIEF:  # VERBOSE, DEBUG
                    warnings.warn(
                        'Incomplete PAX numbers for FLOPS fallout - assume same as design'
                    )
                num_first = self.aviary_inputs.get_val(Aircraft.CrewPayload.Design.NUM_FIRST_CLASS)
                num_business = self.aviary_inputs.get_val(
                    Aircraft.CrewPayload.Design.NUM_BUSINESS_CLASS
                )
                num_tourist = self.aviary_inputs.get_val(
                    Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS
                )
            if wing_cargo is None or misc_cargo is None:
                if verbosity > Verbosity.BRIEF:  # VERBOSE, DEBUG
                    warnings.warn(
                        'Incomplete Cargo masses for FLOPS fallout - assume same as design'
                    )
                wing_cargo = self.aviary_inputs.get_val(Aircraft.CrewPayload.WING_CARGO, 'lbm')
                misc_cargo = self.aviary_inputs.get_val(Aircraft.CrewPayload.MISC_CARGO, 'lbm')
            num_pax = cargo_mass = 0
        elif mass_method == LegacyCode.GASP:
            if num_pax is None:
                if verbosity > Verbosity.BRIEF:  # VERBOSE, DEBUG
                    warnings.warn('Unspecified PAX number for GASP fallout - assume same as design')
                num_pax = self.aviary_inputs.get_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS)
            if cargo_mass is None:
                if verbosity > Verbosity.BRIEF:  # VERBOSE, DEBUG
                    warnings.warn('Unspecified Cargo mass for GASP fallout - assume same as design')
                cargo_mass = self.get_val(Aircraft.CrewPayload.CARGO_MASS, 'lbm')
            num_first = num_business = num_tourist = wing_cargo = misc_cargo = 0

        if phase_info is None:
            phase_info = self.model.phase_info
        if mission_range is None:
            # mission range is sliced from a column vector numpy array, i.e. it is a len
            # 1 numpy array
            mission_range = self.get_val(Mission.Design.RANGE)[0]

        # gross mass is sliced from a column vector numpy array, i.e. it is a len 1 numpy
        # array
        mission_mass = self.get_val(Mission.Design.GROSS_MASS)
        optimizer = self.driver.options['optimizer']

        prob_alternate = _load_off_design(
            json_filename,
            ProblemType.ALTERNATE,
            equations_of_motion,
            mass_method,
            phase_info,
            num_first,
            num_business,
            num_tourist,
            num_pax,
            wing_cargo,
            misc_cargo,
            cargo_mass,
            mission_range,
            mission_mass,
        )

        # TODO: All these methods will need to be updated
        prob_alternate.check_and_preprocess_inputs()
        prob_alternate.add_pre_mission_systems()
        prob_alternate.add_phases()
        prob_alternate.add_post_mission_systems()
        prob_alternate.link_phases()
        prob_alternate.add_driver(optimizer, verbosity=verbosity)
        prob_alternate.add_design_variables()
        prob_alternate.add_objective()
        prob_alternate.setup()
        prob_alternate.set_initial_guesses()
        if run_mission:
            prob_alternate.run_aviary_problem(record_filename='alternate_problem_history.db')
        return prob_alternate

    def fallout_mission(
        self,
        run_mission=True,
        json_filename='sizing_problem.json',
        num_first=None,
        num_business=None,
        num_tourist=None,
        num_pax=None,
        wing_cargo=None,
        misc_cargo=None,
        cargo_mass=None,
        mission_mass=None,
        phase_info=None,
        verbosity=None,
    ):
        """
        This function runs a fallout mission based on a sizing mission output.

        Parameters
        ----------
        run_mission : bool
            Flag to determine whether to run the mission before returning the problem
            object.
        json_filename : str
            Name of the file that the sizing mission has been saved to.
        mission_mass : float, optional
            Takeoff mass for the fallout mission.
        payload_mass : float, optional
            Mass of the payload for the mission.
        phase_info : dict, optional
            Dictionary containing the phases and their required parameters.
        verbosity : Verbosity or int, optional
            Controls the level of printouts for this method. If None, uses the value of
            Settings.VERBOSITY in provided aircraft data.
        """
        # `self.verbosity` is "true" verbosity for entire run. `verbosity` is verbosity
        # override for just this method
        if verbosity is not None:
            # compatibility with being passed int for verbosity
            verbosity = Verbosity(verbosity)
        else:
            verbosity = self.verbosity  # defaults to BRIEF

        mass_method = self.aviary_inputs.get_val(Settings.MASS_METHOD)
        equations_of_motion = self.aviary_inputs.get_val(Settings.EQUATIONS_OF_MOTION)
        if mass_method == LegacyCode.FLOPS:
            if num_first is None or num_business is None or num_tourist is None:
                if verbosity > Verbosity.BRIEF:  # VERBOSE, DEBUG
                    warnings.warn(
                        'Incomplete PAX numbers for FLOPS fallout - assume same as design'
                    )
                num_first = self.aviary_inputs.get_val(Aircraft.CrewPayload.Design.NUM_FIRST_CLASS)
                num_business = self.aviary_inputs.get_val(
                    Aircraft.CrewPayload.Design.NUM_BUSINESS_CLASS
                )
                num_tourist = self.aviary_inputs.get_val(
                    Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS
                )
            if wing_cargo is None or misc_cargo is None:
                if verbosity > Verbosity.BRIEF:  # VERBOSE, DEBUG
                    warnings.warn(
                        'Incomplete Cargo masses for FLOPS fallout - assume same as design'
                    )
                wing_cargo = self.aviary_inputs.get_val(Aircraft.CrewPayload.WING_CARGO, 'lbm')
                misc_cargo = self.aviary_inputs.get_val(Aircraft.CrewPayload.MISC_CARGO, 'lbm')
            num_pax = cargo_mass = 0
        elif mass_method == LegacyCode.GASP:
            if num_pax is None:
                if verbosity > Verbosity.BRIEF:  # VERBOSE, DEBUG
                    warnings.warn('Unspecified PAX number for GASP fallout - assume same as design')
                num_pax = self.aviary_inputs.get_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS)
            if cargo_mass is None:
                if verbosity > Verbosity.BRIEF:  # VERBOSE, DEBUG
                    warnings.warn('Unspecified Cargo mass for GASP fallout - assume same as design')
                cargo_mass = self.get_val(Aircraft.CrewPayload.CARGO_MASS, 'lbm')
            num_first = num_business = num_tourist = wing_cargo = misc_cargo = 0

        if phase_info is None:
            phase_info = self.model.phase_info
        if mission_mass is None:
            # mission mass is sliced from a column vector numpy array, i.e. it is a len 1
            # numpy array
            mission_mass = self.get_val(Mission.Design.GROSS_MASS)[0]

        optimizer = self.driver.options['optimizer']

        prob_fallout = _load_off_design(
            json_filename,
            ProblemType.FALLOUT,
            equations_of_motion,
            mass_method,
            phase_info,
            num_first,
            num_business,
            num_tourist,
            num_pax,
            wing_cargo,
            misc_cargo,
            cargo_mass,
            None,
            mission_mass,
            verbosity=verbosity,
        )

        prob_fallout.check_and_preprocess_inputs()
        prob_fallout.add_pre_mission_systems()
        prob_fallout.add_phases()
        prob_fallout.add_post_mission_systems()
        prob_fallout.link_phases()
        prob_fallout.add_driver(optimizer, verbosity=verbosity)
        prob_fallout.add_design_variables()
        prob_fallout.add_objective()
        prob_fallout.setup()
        prob_fallout.set_initial_guesses()
        if run_mission:
            prob_fallout.run_aviary_problem(record_filename='fallout_problem_history.db')
        return prob_fallout

    def save_sizing_to_json(self, json_filename='sizing_problem.json'):
        """
        This function saves an aviary problem object into a json file.

        Parameters
        ----------
        aviary_problem : AviaryProblem
            Aviary problem object optimized for the aircraft design/sizing mission.
            Assumed to contain aviary_inputs and Mission.Summary.GROSS_MASS
        json_filename : string
            User specified name and relative path of json file to save the data into.
        """
        aviary_input_list = []
        with open(json_filename, 'w') as jsonfile:
            # Loop through aviary input datastructure and create a list
            for data in self.aviary_inputs:
                (name, (value, units)) = data
                type_value = type(value)

                # Get the gross mass value from the sizing problem and add it to input
                # list
                if name == Mission.Summary.GROSS_MASS or name == Mission.Design.GROSS_MASS:
                    Mission_Summary_GROSS_MASS_val = self.get_val(
                        Mission.Summary.GROSS_MASS, units=units
                    )
                    Mission_Summary_GROSS_MASS_val_list = Mission_Summary_GROSS_MASS_val.tolist()
                    value = Mission_Summary_GROSS_MASS_val_list[0]

                else:
                    # there are different data types we need to handle for conversion to
                    # json format
                    # int, bool, float doesn't need anything special

                    # Convert numpy arrays to lists
                    if type_value == np.ndarray:
                        value = value.tolist()

                    # Lists are fine except if they contain enums or Paths
                    if type_value == list:
                        if isinstance(value[0], Enum) or isinstance(value[0], Path):
                            for i in range(len(value)):
                                value[i] = str(value[i])

                    # Enums and Paths need converting to a string
                    if isinstance(value, Enum) or isinstance(value, Path):
                        value = str(value)

                # Append the data to the list
                aviary_input_list.append([name, value, units, str(type_value)])

            # Write the list to a json file
            json.dump(
                aviary_input_list,
                jsonfile,
                sort_keys=True,
                indent=4,
                ensure_ascii=False,
            )

            jsonfile.close()

    def _add_hybrid_objective(self, phase_info):
        phases = list(phase_info.keys())
        takeoff_mass = self.aviary_inputs.get_val(Mission.Design.GROSS_MASS, units='lbm')

        obj_comp = om.ExecComp(
            f'obj = -final_mass / {takeoff_mass} + final_time / 5.',
            final_mass={'units': 'lbm'},
            final_time={'units': 'h'},
        )
        self.model.add_subsystem('obj_comp', obj_comp)

        final_phase_name = phases[-1]
        self.model.connect(
            f'traj.{final_phase_name}.timeseries.mass',
            'obj_comp.final_mass',
            src_indices=[-1],
        )
        self.model.connect(
            f'traj.{final_phase_name}.timeseries.time',
            'obj_comp.final_time',
            src_indices=[-1],
        )

    def _save_to_csv_file(self, filename):
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['name', 'value', 'units']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            for name, value_units in sorted(self.aviary_inputs):
                value, units = value_units
                writer.writerow({'name': name, 'value': value, 'units': units})


def _read_sizing_json(aviary_problem, json_filename):
    """
    This function reads in an aviary problem object from a json file.

    Parameters
    ----------
    aviary_problem: OpenMDAO Aviary Problem
        Aviary problem object optimized for the aircraft design/sizing mission.
        Assumed to contain aviary_inputs and Mission.Summary.GROSS_MASS
    json_filename:   string
        User specified name and relative path of json file to save the data into

    Returns
    -------
    Aviary Problem object with updated input values from json file

    """
    # load saved input list from json file
    with open(json_filename) as json_data_file:
        loaded_aviary_input_list = json.load(json_data_file)
        json_data_file.close()

    # Loop over input list and assign aviary problem input values
    counter = 0  # list index tracker
    for inputs in loaded_aviary_input_list:
        [var_name, var_values, var_units, var_type] = inputs

        # Initialize some flags to identify enums
        is_enum = False

        if var_type == "<class 'list'>":
            # check if the list contains enums
            for i in range(len(var_values)):
                if isinstance(var_values[i], str):
                    if var_values[i].find('<') != -1:
                        # Found a list of enums: set the flag
                        is_enum = True

                        # Manipulate the string to find the value
                        tmp_var_values = var_values[i].split(':')[-1]
                        var_values[i] = (
                            tmp_var_values.replace('>', '')
                            .replace(']', '')
                            .replace("'", '')
                            .replace(' ', '')
                        )

            if is_enum:
                var_values = convert_strings_to_data(var_values)

        elif var_type.find('<enum') != -1:
            # Identify enums and manipulate the string to find the value
            tmp_var_values = var_values.split(':')[-1]
            var_values = (
                tmp_var_values.replace('>', '').replace(']', '').replace("'", '').replace(' ', '')
            )
            var_values = convert_strings_to_data([var_values])

        # Check if the variable is in meta data
        if var_name in BaseMetaData.keys():
            try:
                aviary_problem.aviary_inputs.set_val(
                    var_name, var_values, units=var_units, meta_data=BaseMetaData
                )

            except BaseException:
                # Print helpful warning
                # TODO "FAILURE" implies something more serious, should this be a raised
                # exception?
                warnings.warn(
                    f'FAILURE: list_num = {counter}, Input String = {inputs}, Attempted '
                    f'to set_value({var_name}, {var_values}, {var_units})',
                )
        else:
            # Not in the MetaData
            warnings.warn(
                f'Name not found in MetaData: list_num = {counter}, Input String = '
                f'{inputs}, Attempted set_value({var_name}, {var_values}, {var_units})'
            )

        counter = counter + 1  # increment index tracker
    return aviary_problem


def _load_off_design(
    json_filename,
    problem_type,
    equations_of_motion,
    mass_method,
    phase_info,
    num_first,
    num_business,
    num_tourist,
    num_pax,
    wing_cargo,
    misc_cargo,
    cargo_mass,
    mission_range=None,
    mission_gross_mass=None,
    verbosity=Verbosity.BRIEF,
):
    """
    This function loads a sized aircraft, and sets up an aviary problem
    for a specified off design mission.

    Parameters
    ----------
    json_filename : str
        User specified name and relative path of json file containing the sized aircraft data
    problem_type : ProblemType
        Alternate or Fallout. Alternate requires mission_range input and fallout
        requires mission_fuel input
    equations_of_motion : EquationsOfMotion
        Which equations of motion will be used for the off-design mission
    MassMethod : LegacyCode
        Which legacy code mass method will be used (GASP or FLOPS)
    phase_info : dict
        phase_info dictionary for off-design mission
    num_first : int
        Number of first class passengers on off-design mission (FLOPS only)
    num_business : int
        Number of business class passengers on off-design mission (FLOPS only)
    num_tourist : int
        Number of economy class passengers on off-design mission (FLOPS only)
    num_pax : int
        Total number of passengers on off-design mission (GASP only)
    wing_cargo: float
        Wing-stored cargo mass on off-design mission, in lbm (FLOPS only)
    misc_cargo : float
        Miscellaneous cargo mass on off-design mission, in lbm (FLOPS only)
    cargo_mass : float
        Total cargo mass on off-design mission, in lbm (GASP only)
    mission_range : float
        Total range of off-design mission, in NM
    mission_gross_mass : float
        Aircraft takeoff gross mass for off-design mission, in lbm
    verbosity : Verbosity or list, optional
        Controls the level of printouts for this method.

    Returns
    -------
    Aviary Problem object with completed load_inputs() for specified off design mission
    """
    # Initialize a new aviary problem and aviary_input data structure
    prob = AviaryProblem()
    prob.aviary_inputs = AviaryValues()

    prob = _read_sizing_json(prob, json_filename)

    # Update problem type
    prob.problem_type = problem_type
    prob.aviary_inputs.set_val('settings:problem_type', problem_type)
    prob.aviary_inputs.set_val('settings:equations_of_motion', equations_of_motion)

    # Setup Payload
    if mass_method == LegacyCode.FLOPS:
        prob.aviary_inputs.set_val(
            Aircraft.CrewPayload.NUM_FIRST_CLASS, num_first, units='unitless'
        )
        prob.aviary_inputs.set_val(
            Aircraft.CrewPayload.NUM_BUSINESS_CLASS, num_business, units='unitless'
        )
        prob.aviary_inputs.set_val(
            Aircraft.CrewPayload.NUM_TOURIST_CLASS, num_tourist, units='unitless'
        )
        num_pax = num_first + num_business + num_tourist
        prob.aviary_inputs.set_val(Aircraft.CrewPayload.MISC_CARGO, misc_cargo, 'lbm')
        prob.aviary_inputs.set_val(Aircraft.CrewPayload.WING_CARGO, wing_cargo, 'lbm')
        cargo_mass = misc_cargo + wing_cargo

    prob.aviary_inputs.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, num_pax, units='unitless')
    prob.aviary_inputs.set_val(Aircraft.CrewPayload.CARGO_MASS, cargo_mass, 'lbm')

    if problem_type == ProblemType.ALTERNATE:
        # Set mission range, aviary will calculate required fuel
        if mission_range is None:
            if verbosity > Verbosity.BRIEF:  # VERBOSE, DEBUG
                # TODO text says this is an "ERROR" but methods continues to run, this
                #      might be confusion
                warnings.warn(
                    'ERROR in _load_off_design - Alternate problem type requested with '
                    'no specified Range'
                )
        else:
            prob.aviary_inputs.set_val(Mission.Design.RANGE, mission_range, units='NM')
            prob.aviary_inputs.set_val(Mission.Summary.RANGE, mission_range, units='NM')
            # TODO is there a reason we can't use set_default() to make sure target range exists and
            #      has a value if not already in dictionary?
            try:
                phase_info['post_mission']['target_range']
                phase_info['post_mission']['target_range'] = (mission_range, 'nmi')
            except KeyError:
                warnings.warn('no target range to update')

    elif problem_type == ProblemType.FALLOUT:
        # Set mission fuel and calculate gross weight, aviary will calculate range
        if mission_gross_mass is None:
            if verbosity > Verbosity.BRIEF:  # VERBOSE, DEBUG
                warnings.warn(
                    'Error in _load_off_design - Fallout problem type requested with no '
                    'specified Gross Mass'
                )
        else:
            prob.aviary_inputs.set_val(Mission.Summary.GROSS_MASS, mission_gross_mass, units='lbm')

    # Load inputs
    prob.load_inputs(prob.aviary_inputs, phase_info)
    return prob
