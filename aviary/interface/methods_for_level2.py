import csv
import json
import os
import warnings
from copy import deepcopy
from datetime import datetime
from enum import Enum
from pathlib import Path

import dymos as dm
import numpy as np
import openmdao
import openmdao.api as om
from openmdao.utils.reports_system import _default_reports
from openmdao.utils.units import convert_units
from packaging import version

from aviary.core.aviary_group import AviaryGroup
from aviary.interface.utils import set_warning_format
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.csv_data_file import write_data_file
from aviary.utils.functions import convert_strings_to_data, get_path
from aviary.utils.merge_variable_metadata import merge_meta_data
from aviary.utils.named_values import NamedValues
from aviary.variable_info.enums import EquationsOfMotion, LegacyCode, ProblemType, Verbosity
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

    def __init__(
        self,
        problem_type: ProblemType = None,
        verbosity=None,
        meta_data=BaseMetaData.copy(),
        **kwargs,
    ):
        # Modify OpenMDAO's default_reports for this session.
        new_reports = [
            'subsystems',
            'mission',
            'timeseries_csv',
            'run_status',
            'sizing_results',
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

        self.problem_type = problem_type
        if problem_type == ProblemType.MULTI_MISSION:
            self.model = om.Group()
        else:
            self.model = AviaryGroup()
            self.aviary_inputs = None

        self.aviary_groups_dict = {}

        self.meta_data = meta_data

        # TODO try and find a better solution than a new custom flag - the issue is multimission
        #      problems don't have a consistent variable path to check the inputs later on
        self.generate_payload_range = False

    def load_inputs(
        self,
        aircraft_data,
        phase_info=None,
        engine_builders=None,
        problem_configurator=None,
        meta_data=None,
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

        if meta_data is not None:
            # Support for custom meta_data set.
            self.meta_data = meta_data

        # TODO: We cannot pass self.verbosity back up from load inputs for multi-mission because there could be multiple .csv files
        self.model.meta_data = self.meta_data
        aviary_inputs, verbosity = self.model.load_inputs(
            aircraft_data=aircraft_data,
            phase_info=phase_info,
            engine_builders=engine_builders,
            problem_configurator=problem_configurator,
            verbosity=verbosity,
        )

        self.aviary_inputs = aviary_inputs
        self.verbosity = verbosity
        if self.problem_type is None:
            # if there are multiple load_inputs() calls, only the problem type from the first aviary_values is used
            self.problem_type = aviary_inputs.get_val(Settings.PROBLEM_TYPE)

        # TODO try and find a better solution than a new custom flag - the issue is multimission
        #      problems don't have a consistent variable path to check the inputs later on
        # BUG you can't specify generating payload-range diagram via aviary_inputs after load_inputs
        if Settings.PAYLOAD_RANGE in aviary_inputs:
            self.generate_payload_range = aviary_inputs.get_val(Settings.PAYLOAD_RANGE)

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

        # we have to update meta data after check_and_preprocess because metadata update
        # requires get_all_subsystems, which requires core_subsystems, which doesn't exist until
        # after check_and_preprocess is assembled
        self._update_metadata_from_subsystems(self.model)  # update meta data with new entries

    def _update_metadata_from_subsystems(self, group):
        """Merge metadata from user-defined subsystems into problem metadata."""
        # loop through phase_info and external subsystems
        for phase_name in group.mission_info:
            external_subsystems = group.get_all_subsystems(
                group.mission_info[phase_name]['external_subsystems']
            )

            for subsystem in external_subsystems:
                meta_data = subsystem.meta_data.copy()
                self.meta_data = merge_meta_data([self.meta_data, meta_data])

        # Update the reference to the newly merged meta_data.
        group.meta_data = self.meta_data

    def add_aviary_group(
        self,
        name: str,
        aircraft: AviaryValues,
        mission: dict,
        engine_builders=None,
        problem_configurator=None,
        verbosity: Verbosity = Verbosity.BRIEF,
    ):
        """
        Used for creating a multi-mission problem. This method creates an AviaryGroup() for each
        airraft and mission combination. It can also accept a specific engine_builder for each
        group. The method loads and checks_and_preprocesses inputs, and then combines metadata.

        Parameters
        ----------
        name : string
            A unique name that identifies this group which can be referenced later.
        aircraft : AviaryValues object
            Defines the aircraft configuration
        mission : phase_info, dict
            Defines the mission the aircraft will fly
        engine_builders : EngineBuilder object, optional
            Defines a custom engine model
        problem_configurator ; ProblemConfigurator, optional
            Required when setting custom equations of motion. See two_dof_problem_configurator.py for an example.
        verbosity : Verbosity or int, optional
            Controls the level of printouts for this method.

        Returns
        -------
        subsystem
            The AviaryGroup object containing the specified aircraft, mission, and engine model.

        """
        if self.problem_type is not ProblemType.MULTI_MISSION:
            ValueError(
                'add_aviary_group() should only be called when ProblemType is MULTI_MISSION.'
            )

        sub = self.model.add_subsystem(name, AviaryGroup())
        sub.meta_data = self.meta_data
        sub.load_inputs(
            aircraft_data=aircraft,
            phase_info=mission,
            engine_builders=engine_builders,
            problem_configurator=problem_configurator,
            verbosity=verbosity,
        )

        sub.check_and_preprocess_inputs()

        self.aviary_groups_dict[name] = sub

        if self.verbosity is None:
            # If problem-level verbosity was not defined, use the verbosity specified in the first
            # added AviaryGroup
            self.verbosity = sub.verbosity

        # TODO try and find a better solution than a new custom flag - the issue is multimission
        #      problems don't have a consistent variable path to check the inputs later on
        if Settings.PAYLOAD_RANGE in sub.aviary_inputs:
            self.generate_payload_range = sub.aviary_inputs.get_val(Settings.PAYLOAD_RANGE)

        self._update_metadata_from_subsystems(sub)  # update meta data with new entries

        return sub

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
        """
        # `self.verbosity` is "true" verbosity for entire run. `verbosity` is verbosity
        # override for just this method
        if verbosity is not None:
            # compatibility with being passed int for verbosity
            verbosity = Verbosity(verbosity)
        else:
            verbosity = self.verbosity  # defaults to BRIEF

        if self.problem_type == ProblemType.MULTI_MISSION:
            for name, group in self.aviary_groups_dict.items():
                group.add_pre_mission_systems(verbosity=verbosity)
        else:
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

        if self.problem_type == ProblemType.MULTI_MISSION:
            for name, group in self.aviary_groups_dict.items():
                Traj = group.add_phases(
                    phase_info_parameterization=phase_info_parameterization,
                    parallel_phases=parallel_phases,
                    verbosity=verbosity,
                    comm=self.comm,
                )
        else:
            Traj = self.model.add_phases(
                phase_info_parameterization=phase_info_parameterization,
                parallel_phases=parallel_phases,
                verbosity=verbosity,
                comm=self.comm,
            )

        return Traj

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
        """
        # `self.verbosity` is "true" verbosity for entire run. `verbosity` is verbosity
        # override for just this method
        if verbosity is not None:
            # compatibility with being passed int for verbosity
            verbosity = Verbosity(verbosity)
        else:
            verbosity = self.verbosity  # defaults to BRIEF

        if self.problem_type == ProblemType.MULTI_MISSION:
            for name, group in self.aviary_groups_dict.items():
                group.add_post_mission_systems(verbosity=verbosity)
        else:
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

        if self.problem_type == ProblemType.MULTI_MISSION:
            for name, group in self.aviary_groups_dict.items():
                group.link_phases(verbosity=verbosity, comm=self.comm)
        else:
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
            driver.opt_settings['Major feasibility tolerance'] = 1e-6

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

        if self.problem_type == ProblemType.MULTI_MISSION:
            for name, group in self.aviary_groups_dict.items():
                group.add_design_variables(problem_type=self.problem_type, verbosity=verbosity)
        else:
            self.model.add_design_variables(problem_type=self.problem_type, verbosity=verbosity)

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
                self._add_hybrid_objective(self.model.mission_info)
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

            if self.problem_type is ProblemType.SIZING:
                self.model.add_objective(Mission.Objectives.FUEL, ref=ref)

            elif self.problem_type is ProblemType.ALTERNATE:
                self.model.add_objective(Mission.Objectives.FUEL, ref=ref)

            elif self.problem_type is ProblemType.FALLOUT:
                # if ref > 0:
                #    # Maximize range.
                #    ref = -ref
                self.model.add_objective(Mission.Objectives.RANGE, ref=ref)

            else:
                raise ValueError(f'{self.problem_type} is not a valid problem type.')

    def add_design_var_default(
        self,
        name: str,
        lower: float = None,
        upper: float = None,
        units: str = None,
        src_shape=None,
        default_val: float = None,
        ref: float = None,
    ):
        """
        Add a design variable to the problem as well as initialized a default value for that design variable.

        The default value can be over-written after setup with prob.set_val()

        Parameters
        ----------
        name : string
            A unique name that identifies this design variable.
        lower : float, optional
            The lowest value that the optimizer can choose for this design variable.
        upper : float, optional
            The largest value that the optimizer can choose for this design variable.
        src_shape : int or tuple, optional
            Assumed shape of any connected source or higher level promoted input.
        default_val : float or ndarray, optional
            The default value to be assigned to this design variable.
        ref : float or ndarray, optional
            Value of design var that scales to 1.0 in the driver.

        """
        self.model.add_design_var(name=name, lower=lower, upper=upper, units=units, ref=ref)
        if default_val is not None:
            self.model.set_input_defaults(
                name=name, val=default_val, units=units, src_shape=src_shape
            )

    def set_design_range(self, missions: list[str], range: str):
        # TODO: What happens if design range is specified in CSV??? should be able to access from group.aviary_values
        """
        Used for multi-mission problems. This method finds the longest mission and sets its range
        as the design range for all AviaryGroups. design_range is used within Aviary for sizing
        subsystems (avionics and AC). This could be simpllified in the future if there was a
        single pre-mission for similar aircraft.

        Parameters
        ----------
        missions : list[str]
            The names of all the missions instantiated via add_aviary_group()
        range : str
            The promoted name of the range variable. i.e. "Aircraft1.Range"

        """
        matching_names = [
            (name, group) for name, group in self.aviary_groups_dict.items() if name in missions
        ]
        design_range = []
        # loop through all the phase_info and extract target ranges
        for name, group in matching_names:
            target_range, units = group.post_mission_info['target_range']
            design_range.append(convert_units(target_range, units, 'nmi'))
        # TODO: loop through all the .csv files and extract Mission.Design.RANGE
        design_range_max = np.max(design_range)
        self.set_val(range, val=design_range_max, units='nmi')

    def add_composite_objective(self, *args, ref: float = None):
        """
        Creates composite_objective output by assemblin an ExecComp based on a variety of AviaryGroup
        outputs selected by the user. A number of different outputs from the same or different
        aricraft can be combined in this way such as creating an objective function based on fuel
        plus noise emissions. Each objective output should include a weight otherwise the weight will be
        assumed to be equal (i.e. fuel is equally important as reducing noise emissions).

        Parameters
        ----------
        *args : a list of 3-tuple, 2-tuple, str. Or it can be left empty
            If 3-tuple: (model, output, weight)
            If 2-tuple: (model, output) or (output, weight)
            If 1-tuple: (output) or 'fuel', 'fuel_burned', 'mass', 'range', 'time'
            If empty, information will be populated based on problem_type:
            - If ProblemType = FALLOUT, objective = Mission.Objectives.RANGE
            - If ProblemType = Sizing or Alternate, objective = Mission.Objectives.FUEL

            Example inputs can be any of the following:
            ('fuel')
            (Mission.Summary.FUEL_BURNED)
            (Mission.Summary.FUEL_BURNED, Mission.Summary.CO2)
            ('model1', Mission.Summary.FUEL_BURNED)
            (Mission.Summary.FUEL_BURNED, 1.0)
            (Mission.Summary.FUEL_BURNED, 1.0), (Mission.Summary.CO2, 2.0)
            ('model1', Mission.Summary.FUEL_BURNED), ('model2', Mission.Summary.CO2)
            ('model1', Mission.Summary.FUEL_BURNED, 1.0), ('model2', Mission.Summary.CO2, 2.0)

        ref : float, optional
            Reference value for the final objective for scaling.

        Behavior
        --------
        - Connects each specified mission output into a newly created `ExecComp` block.
        - Computes a weighted sum: each output is weighted by both the total weights
        - Adds the result as the final objective named `'composite_objective'`, accessible at the top level model.
        """

        # There are LOTS of different ways for the users to input str, 2-tuple, or 3-tuple into *args
        # Correct combinations are (output), (output, weight), (model, output), or (model, output, weight).
        # We have to catch every case and advise the user on how to corect their errors and add defaults as needed.
        default_model = 'model'
        default_weight = 1.0
        objectives = []
        for arg in args:
            if isinstance(arg, tuple) and len(arg) == 3:
                model, output, weight = arg
                if model not in self.aviary_groups_dict:
                    raise ValueError(
                        f'The first element specified in {arg} must be the model name.'
                    )
            elif isinstance(arg, tuple) and len(arg) == 2:
                first, second = arg
                if isinstance(first, str) and isinstance(second, str):
                    if first in self.aviary_groups_dict:
                        # we have the model and output but no weight
                        model, output, weight = first, second, default_weight
                    else:
                        raise ValueError(
                            f'The first element specified in {arg} must be the model name.'
                        )
                elif isinstance(first, str) and isinstance(second, (float, int)):
                    if first in self.aviary_groups_dict:
                        raise ValueError(
                            f'When specifying {arg}, the user specified a model name and a weight '
                            f'but did not specify what output from that model the weight should be applied to.'
                        )
                    else:
                        # we have the output and the weight but not model
                        model, output, weight = default_model, first, second
                else:
                    raise ValueError(
                        f'The user specified {arg} which is not a 2-tuple of (model, output) or (output, weight).'
                    )
            elif isinstance(arg, str):
                if arg in self.aviary_groups_dict:
                    raise ValueError(
                        f"When specifying '{arg}', the user provided only a model name "
                        f'but did not specify what output from that model should be used as the objective.'
                    )
                else:
                    # we have an output and we use the default model and weights
                    model, output, weight = default_model, arg, default_weight

            # in some cases the users provides no input and we can derive the objectie from the problem type:
            elif self.model.problem_type is ProblemType.SIZING:
                model, output, weight = default_model, Mission.Objectives.FUEL, default_weight
            elif self.model.problem_type is ProblemType.ALTERNATE:
                model, output, weight = default_model, Mission.Objectives.FUEL, default_weight
            elif self.model.problem_type is ProblemType.FALLOUT:
                model, output, weight = default_model, Mission.Objectives.RANGE, default_weight
            else:
                raise ValueError(
                    f'Unrecognized objective format: {arg}. '
                    f'Each argument must be one of the following: '
                    f'(output), (output, weight), (model, output), or (model, output, weight).'
                    f'Outputs can be from the variable meta data, or can be: fuel_burned, fuel'
                    f'Or problem type must be set to SIZING, ALTERNATE, or FALLOUT'
                )
            objectives.append((model, output, weight))
            # objectives = [
            # ('model1', Mission.Summary.FUEL_BURNED, 1),
            # ('model2', Mission.Summary.CO2, 1),
            #  ...
            # ]

        # Dictionary for default reference values
        default_ref_values = {
            'mass': -5e4,
            'hybrid_objective': -5e4,
            'fuel_burned': 1e4,
            'fuel': 1e4,
        }

        # Now checkout the output and see if we have recognizable strings and replace them with the variable meta data name
        objectives_cleaned = []
        for model, output, weight in objectives:
            if output == 'fuel_burned':
                output = Mission.Summary.FUEL_BURNED
                # default scaling is valid only if this is the only argument and the ref has not yet been set
                if len(args) == 1 and ref == None:
                    # set a default ref
                    ref = default_ref_values['fuel_burned']
            elif output == 'fuel':
                output = Mission.Objectives.FUEL
                if len(args) == 1 and ref == None:
                    ref = default_ref_values['fuel']
            elif output == 'mass':
                output = Mission.Summary.FINAL_MASS
                if len(args) == 1 and ref == None:
                    ref = default_ref_values['mass']
            elif output == 'time':
                output = Mission.Summary.FINAL_TIME
            elif output == 'range':
                output = Mission.Summary.RANGE  # Unsure if this will work
            objectives_cleaned.append((model, output, weight))

        # Create the calculation string for the ExecComp() and the promotion reference values
        weighted_exprs = []
        connection_names = []
        obj_inputs = []
        total_weight = sum(weight for _, _, weight in objectives_cleaned)
        for model, output, weight in objectives_cleaned:
            output_safe = output.replace(':', '_')

            # we use "_" here because ExecComp() cannot intake "."
            obj_input = f'{model}_{output_safe}'
            obj_inputs.append(obj_input)
            weighted_exprs.append(f'{obj_input}*{weight}/{total_weight}')
            connection_names.append(
                [f'{model}.{output}', f'composite_function.{model}_{output_safe}']
            )
        final_expr = ' + '.join(weighted_exprs)

        # weighted_str looks like:  'model1_fuelburn*0.67*0.5 + model1_gross_mass*0.33*0.5 + model2_fuelburn*0.67*0.5 + model2_gross_mass*0.33*0.5'

        kwargs = {}
        if version.parse(openmdao.__version__) >= version.parse('3.40'):
            # We can get the correct unit from the source. This prevents a warning.
            kwargs = {k: {'units_by_conn': True} for k in obj_inputs}

        # adding composite execComp to super problem
        self.model.add_subsystem(
            'composite_function',
            om.ExecComp('composite_objective = ' + final_expr, **kwargs),
            promotes_outputs=['composite_objective'],
        )

        # connect from inside of the models to the composite objective
        for source, target in connection_names:
            self.model.connect(source, target)
        # finally add the objective
        self.model.add_objective('composite_objective', ref=ref)

    def add_composite_objective_adv(
        self,
        missions: list[str],
        outputs: list[str],
        mission_weights: list[float] = None,
        output_weights: list[float] = None,
        ref: float = 1.0,
    ):
        """
        Adds a composite objective function to the OpenMDAO problem by aggregating output values across
        multiple mission models, with independent weighting for both missions and outputs. This is most
        useful when you have historical information on how often a given mission was flown (mission_weights)
        and then you have a duel set of objectives you wish to include i.e. for each flight minimize
        both fuel_burned and gross_mass. How important fuel_burned is vs. gross_mass is determined via
        specifying output_weights.

        Parameters
        ----------
        missions : list of str
            List of subsystem names (e.g., 'model1', 'model2') corresponding to different missions.

        outputs : list of str
            List of output variable names (e.g., Mission.Summary.FUEL_BURNED, Mission.Summary.GROSS_MASS) to be included
            in the objective from each mission.

        mission_weights : list of float, optional
            Weights assigned to each mission. If None, equal weighting is assumed.
            These weights will be normalized internally to sum to 1.0.

        output_weights : list of float, optional
            Weights assigned to each output variable. If None, equal weighting is assumed.
            These weights will also be normalized internally to sum to 1.0.

        ref : float, optional
            Reference value for the final objective. Passed to `add_objective()` for scaling.

        Behavior
        --------
        - Connects each specified mission output into a newly created `ExecComp` block.
        - Computes a weighted sum: each output is weighted by both its output weight
        and the weight of the mission it came from.
        - Adds the result as the final objective named `'composite_objective'`, accessible at the top level model.
        """

        # Setup mission and output lengths if they are not already given
        if mission_weights is None:
            mission_weights = np.ones(len(missions))

        if output_weights is None:
            output_weights = np.ones(len(outputs))

        # # Make an ExecComp
        # for mission in missions:
        #     for output in outputs:

        # weights are normalized - e.g. for given weights 3:1, the normalized
        # weights are 0.75:0.25
        # TODO: Remove before push
        # output_weights = [2,1]
        # mission_weights = [1,1]
        # missions = ['model1','model2']
        # outputs = ['fuelburn','gross_mass']
        weighted_exprs = []
        connection_names = []
        output_weights = [float(weight / sum(output_weights)) for weight in output_weights]
        mission_weights = [float(weight / sum(mission_weights)) for weight in mission_weights]
        for mission, mission_weight in zip(missions, mission_weights):
            for output, output_weight in zip(outputs, output_weights):
                connection_names.append(
                    [f'composite_function.{mission}_{output}', f'{mission}.{output}']
                )
                weighted_exprs.append(f'{mission}_{output}*{output_weight}*{mission_weight}')
        final_expr = ' + '.join(weighted_exprs)
        # weighted_str looks like:  'model1.fuelburn*0.67*0.5 + model1.gross_mass*0.33*0.5 + model2.fuelburn*0.67*0.5 + model2.gross_mass*0.33*0.5'

        # adding composite execComp to super problem
        self.model.add_subsystem(
            'composite_function',
            om.ExecComp('composite_objective = ' + final_expr, has_diag_partials=True),
            promotes_outputs=['composite_objective'],
        )
        # connect from inside of the models to the composite objective
        for target, source in connection_names:
            self.model.connect(target, source)
        # finally add the objective
        self.model.add_objective('composite_objective', ref=ref)

    def build_model(self, verbosity=None):
        """
        A lightly wrapped add_pre_mission_systems(), add_phases(), add_post_mission_systems(), and link_phases()
        method to decrease code length for the avarage user. If the user needs finer control, they should
        not use build_model but instead call the four individual methods separately.

        Parameters
        ----------
        verbosity : Verbosity or int, optional
            Controls the level of printouts for this method.
        """
        # `self.verbosity` is "true" verbosity for entire run. `verbosity` is verbosity
        # override for just this method
        if verbosity is not None:
            # compatibility with being passed int for verbosity
            verbosity = Verbosity(verbosity)
        else:
            verbosity = self.verbosity  # defaults to BRIEF

        if self.problem_type == ProblemType.MULTI_MISSION:
            for name, group in self.aviary_groups_dict.items():
                group.add_pre_mission_systems(verbosity=verbosity)
                group.add_phases(verbosity=verbosity, comm=self.comm)
                group.add_post_mission_systems(verbosity=verbosity)
                group.link_phases(verbosity=verbosity, comm=self.comm)
        else:
            self.model.add_pre_mission_systems(verbosity=verbosity)
            self.model.add_phases(verbosity=verbosity, comm=self.comm)
            self.model.add_post_mission_systems(verbosity=verbosity)
            self.model.link_phases(verbosity=verbosity, comm=self.comm)

    def promote_inputs(self, mission_names: list[str], var_pairs: list[tuple[str, str]]):
        """
        Link a promoted input to multiple groups' unpromoted inputs using an internal IVC.

        Parameters
        ----------
        self : om.Problem
            The Problem instance this is being called from.

        missions : list of str
            The subsystem names receiving the connection.

        var_pairs : list of (str, str)
            Each pair is (input_name_in_group, top_level_name_to_use)
        """

        #
        for name, group in self.aviary_groups_dict.items():
            for mission_name in mission_names:
                if name == mission_name:
                    # the group name matches the mission name,
                    # group.promotes(var_pairs)
                    # print("var_pairs",var_pairs)
                    self.model.promotes(mission_name, inputs=var_pairs)

    def setup(self, **kwargs):
        """
        A lightly wrapped setup() and set_initial_defaults() method for the problem.

        Parameters
        ----------
        verbosity : Verbosity or int, optional
            Controls the level of printouts for this method.
        **kwargs : optional
            All arguments to be passed to the OpenMDAO prob.setup() method.
        """
        # verbosity is not used in this method, but it is understandable that a user
        # might try and include it (only method that doesn't accept it). Capture it
        if 'verbosity' in kwargs:
            kwargs.pop('verbosity')
        # Use OpenMDAO's model options to pass all options through the system hierarchy.

        if self.problem_type == ProblemType.MULTI_MISSION:
            for name, group in self.aviary_groups_dict.items():
                setup_model_options(
                    self, group.aviary_inputs, group.meta_data, prefix=name, group=group
                )
                with warnings.catch_warnings():
                    # group.aviary_inputs is already set
                    group.meta_data = self.meta_data  # <- meta_data is the same for all groups
                    # group.phase_info is already set
        else:
            setup_model_options(self, self.aviary_inputs, self.meta_data)
            # suppress warnings:
            # "input variable '...' promoted using '*' was already promoted using 'aircraft:*'
            with warnings.catch_warnings():
                self.model.aviary_inputs = (
                    self.aviary_inputs
                )  # <- there is only one aviary_inputs in this case
                self.model.meta_data = self.meta_data
                # self.model.phase_info is already set

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', om.OpenMDAOWarning)
            warnings.simplefilter('ignore', om.PromotionWarning)

            super().setup(**kwargs)

        self.set_initial_guesses(verbosity=None)

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

        if self.problem_type == ProblemType.MULTI_MISSION:
            for name, group in self.aviary_groups_dict.items():
                group.set_initial_guesses(
                    parent_prob=parent_prob,
                    parent_prefix=parent_prefix,
                    verbosity=verbosity,
                )

        else:
            self.model.set_initial_guesses(
                parent_prob=parent_prob,
                parent_prefix=parent_prefix,
                verbosity=verbosity,
            )

    def run_aviary_problem(
        self,
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
        verbosity : Verbosity or int, optional
            Controls the level of printouts for this method.
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
            with open(self.get_reports_dir() / 'input_list.txt', 'w') as outfile:
                self.model.list_inputs(out_stream=outfile)

            recorder = om.SqliteRecorder('optimization_history.db')
            self.driver.add_recorder(recorder)

        if suppress_solver_print:
            self.set_solver_print(level=0)

        # and run mission, and dynamics
        if run_driver:
            self.result = dm.run_problem(
                self,
                run_driver=run_driver,
                simulate=simulate,
                make_plots=make_plots,
                solution_record_file='problem_history.db',
                restart=restart_filename,
            )

            # Manually print out a failure message for low verbosity modes that suppress
            # optimizer printouts, which may include the results message. Assumes success,
            # alerts user on a failure
            if (
                not self.result.success and verbosity <= Verbosity.BRIEF  # QUIET, BRIEF
            ):
                warnings.warn('\nAviary run failed. See the dashboard for more details.\n')
        else:
            self.run_model()
            self.result = self.driver.result

        # update n2 diagram after run.
        outdir = Path(self.get_reports_dir(force=True))
        outfile = os.path.join(outdir, 'n2.html')
        om.n2(
            self,
            outfile=outfile,
            show_browser=False,
        )

        if verbosity >= Verbosity.VERBOSE:  # VERBOSE, DEBUG
            with open(Path(self.get_reports_dir()) / 'output_list.txt', 'w') as outfile:
                self.model.list_outputs(out_stream=outfile)

        if self.generate_payload_range:
            self.run_payload_range()

    def run_off_design_mission(
        self,
        problem_type: ProblemType,
        phase_info=None,
        equations_of_motion: EquationsOfMotion = None,
        problem_configurator=None,
        num_first_class=None,
        num_business=None,
        num_tourist=None,
        num_pax=None,
        wing_cargo=None,
        misc_cargo=None,
        cargo_mass=None,
        mission_gross_mass=None,
        mission_range=None,
        optimizer=None,
        name=None,
        fill_cargo=False,
        fill_fuel=False,
        verbosity=None,
        payload_range_controls=None,
    ):
        """
        Runs the aircraft model in a off-design mission of the specified type. It is assumed that
        the AviaryProblem is loaded with an already sized aircraft.

        Parameters
        ----------
        problem_type : str, ProblemType
            The type of off-design mission to be flown. SIZING missions are not allowed.
        phase_info : dict (optional)
            The phase_info to use for the off-design mission. If not provided, the phase info used
            for the previous Aviary run (typically the design mission) is used.
        equation_of_motion : str, EquationsOfMotion
            Which equations of motion to use for the off-design mission. If not provided, the
            equations of motion used for the previous Aviary run (typically the design mission) is
            used.
        problem_configurator : ProblemConfigurator
            Problem configurator to use for the off-design mission. If not provided, the problem
            configurator used for the previous Aviary run (typically the design mission) is used.
        num_first_class : int, optional
            [FLOPS mass only] Number of first-class passengers flying on the off-design mission.
        num_business : int, optional
            [FLOPS mass only] Number of business-class passengers flying on the off-design mission.
        num_tourist : int, optional
            [FLOPS mass only] Number of tourist-class passengers flying on the off-design mission.
        num_pax : int, optional
            Total number of passengers flying on the off-design mission. Optional if using
            FLOPS-based mass and passengers per class are defined instead.
        wing_cargo : float, optional
            [FLOPS mass only] Mass of wing cargo flying on off-design mission, in pounds-mass.
        misc_cargo : float, optional
            [FLOPS mass only] Mass of miscellaneous cargo flying on off-design mission, in
            pounds-mass.
        cargo_mass : float, optional
            Total cargo mass flying on off-design mission, in pounds-mass. Optional if using FLOPS-
            based mass, individual wing and/or misc cargo is defined, and no additional cargo is
            being carried elsewhere.
        mission_gross_mass : float, optional
            Gross mass of aircraft flying off-design mission, in pounds-mass. Defaults to design
            gross mass. For missions where mass is solved for (such as ALTERNATE missions), this is
            the initial guess.
        mission_range : float, optional
            [ALTERNATE missions only]
            Sets fixed range of flown off-design mission, in nautical miles. Unused for other
            mission types.
        optimizer : string, optional
            Set which optimizer to use for the off-design mission. If not provided, the optimizer
            used for the previously ran sizing mission is used. If that cannot be found, such as
            when a problem is loaded from a json output file, then the default optimizer (SLSQP)
            is chosen.
        name : str, optional
            Name of the off-design problem. Defaults to "{original problem name}_off_design".
        fill_cargo : bool, optional
            Adds a design variable to vary cargo mass to exactly fill the aircraft to design
            takeoff gross weight. Useful for cases where precise cargo mass required is not known,
            or when operating mass can change between missions. Defaults to False. Cannot be used at
            the same time as fill_fuel.
        fill_fuel : bool, optional
            Adds takeoff gross mass as a design variable. Useful for cases when operating mass can
            change between missions. Defaults to False. Cannot be used at the same time as
            fill_cargo.
        verbosity : int, Verbosity
            Sets the printout level for the entire off-design problem that is ran.
        payload_range_controls : bool
            Flag used by run_payload_range method call. Adds a cargo variable as a design variable
            (chosen based on the specific problem), which is allowed to float a small amount to
            account for issues when hardcoding payload & fuel mass for certain points on the
            payload-range diagram. This argument is generally not needed for users manually running
            off-design missions.
        """
        # For off-design missions, provided verbosity will be used for all L2 method calls
        if verbosity is not None:
            # compatibility with being passed int for verbosity
            verbosity = Verbosity(verbosity)
        else:
            verbosity = self.verbosity  # defaults to BRIEF

        # accept str for problem type
        problem_type = ProblemType(problem_type)
        if problem_type is ProblemType.SIZING:
            raise UserWarning('Off-design missions cannot be SIZING missions.')

        if fill_cargo and fill_fuel:
            raise UserWarning(
                'Cannot run an off-design mission with both "fill_cargo" and "fill_fuel" flags '
                'active.'
            )

        if name is None:
            name = name = self._name + '_off_design'
        off_design_prob = AviaryProblem(name=name)

        # Set up problem for mission, such as equations of motion, configurators, etc.
        inputs = deepcopy(self.aviary_inputs)

        design_gross_mass = self.get_val(Mission.Design.GROSS_MASS, units='lbm')[0]
        inputs.set_val(Mission.Design.GROSS_MASS, design_gross_mass, units='lbm')

        if problem_type is not None:
            inputs.set_val(Settings.PROBLEM_TYPE, problem_type)
        if equations_of_motion is not None:
            inputs.set_val(Settings.EQUATIONS_OF_MOTION, equations_of_motion)

        if problem_configurator is not None:
            off_design_prob.model.configurator = problem_configurator

        if phase_info is None:
            # model phase_info only contains mission information, recreate the whole thing here
            phase_info = self.model.mission_info.copy()
            phase_info['pre_mission'] = self.model.pre_mission_info.copy()
            phase_info['post_mission'] = self.model.post_mission_info.copy()

        # update passenger count and cargo masses
        mass_method = inputs.get_val(Settings.MASS_METHOD)

        # only FLOPS cares about seat class or specific cargo categories
        if mass_method == LegacyCode.FLOPS:
            if num_first_class is not None:
                inputs.set_val(Aircraft.CrewPayload.NUM_FIRST_CLASS, num_first_class)
            if num_business is not None:
                inputs.set_val(Aircraft.CrewPayload.NUM_BUSINESS_CLASS, num_business)
            if num_tourist is not None:
                inputs.set_val(Aircraft.CrewPayload.NUM_TOURIST_CLASS, num_tourist)

            if wing_cargo is not None:
                inputs.set_val(Aircraft.CrewPayload.WING_CARGO, wing_cargo, 'lbm')
            if misc_cargo is not None:
                inputs.set_val(Aircraft.CrewPayload.MISC_CARGO, misc_cargo, 'lbm')
        else:
            warnings.warn(
                'Off-design functionality is in beta for GASP-mass based aircraft. Please manually '
                'verify your results.'
            )

        if num_pax is not None:
            inputs.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, num_pax)
        if cargo_mass is not None:
            inputs.set_val(Aircraft.CrewPayload.CARGO_MASS, cargo_mass, 'lbm')

        # NOTE once load_inputs is run, phase info details are stored in prob.model.configurator,
        #      meaning any phase_info changes that happen after load inputs is ignored

        if problem_type is ProblemType.ALTERNATE:
            # Set mission range, aviary will calculate required fuel
            if mission_range is None:
                if verbosity >= Verbosity.VERBOSE:
                    warnings.warn(
                        'Alternate problem type requested with no specified range. Using design '
                        'mission range for the off-design mission.'
                    )
                mission_range = self.get_val(Mission.Summary.RANGE, units='NM')[0]

            phase_info['post_mission']['target_range'] = (
                mission_range,
                'nmi',
            )

        # reset the AviaryProblem to run the new mission
        off_design_prob.load_inputs(inputs, phase_info, verbosity=verbosity)

        # Update inputs that are specific to problem type
        # Some Alternate problem changes had to happen before load_inputs, all fallout problem
        # changes must come after load_inputs
        if problem_type is ProblemType.ALTERNATE:
            off_design_prob.aviary_inputs.set_val(Mission.Summary.RANGE, mission_range, units='NM')
            # set initial guess for Mission.Summary.GROSS_MASS to help optimizer with new design
            # variable bounds.
            if mission_gross_mass is None:
                mission_gross_mass = off_design_prob.aviary_inputs.get_val(
                    Mission.Design.GROSS_MASS, 'lbm'
                )
            off_design_prob.aviary_inputs.set_val(
                Mission.Summary.GROSS_MASS, mission_gross_mass * 0.9, units='lbm'
            )

        elif problem_type is ProblemType.FALLOUT:
            # Set mission fuel and calculate gross weight, aviary will calculate range
            if mission_gross_mass is None:
                if verbosity >= Verbosity.VERBOSE:
                    warnings.warn(
                        'Fallout problem type requested with no specified gross mass. Using design '
                        'takeoff gross mass for the off-design mission.'
                    )
                mission_gross_mass = self.get_val(Mission.Design.GROSS_MASS, units='lbm')[0]

            off_design_prob.aviary_inputs.set_val(
                Mission.Summary.GROSS_MASS, mission_gross_mass, units='lbm'
            )

        off_design_prob.check_and_preprocess_inputs(verbosity=verbosity)
        off_design_prob.add_pre_mission_systems(verbosity=verbosity)
        off_design_prob.add_phases(verbosity=verbosity)
        off_design_prob.add_post_mission_systems(verbosity=verbosity)
        off_design_prob.link_phases(verbosity=verbosity)
        if optimizer is None:
            try:
                optimizer = self.driver.options['optimizer']
            except KeyError:
                optimizer = None
        off_design_prob.add_driver(optimizer, verbosity=verbosity)
        off_design_prob.add_design_variables(verbosity=verbosity)

        # Handle edge case for payload-range diagrams
        # Select which cargo variable makes the most sense to float, and then set a tolerance
        # based on rough guesses on what is sufficient to get the problem to converge without
        # setting design variable bounds too large
        if fill_cargo:
            # GASP cargo mass is an input, can directly use as control variable
            if mass_method is GASP:
                control_var = Aircraft.CrewPayload.CARGO_MASS
                val = cargo_mass
                tol = 1.05
            # FLOPS cargo mass is an output, not valid for control variable. Pick control var.
            else:
                # See if misc_cargo is being used, float that as a backup
                if misc_cargo is None or misc_cargo == 0:
                    # We aren't using cargo_mass OR misc_mass - try wing cargo as last resort
                    if wing_cargo is None or wing_cargo == 0:
                        # We don't know enough about the aircraft to make any informed guesses. Use
                        # arbitrary values
                        control_var = Aircraft.CrewPayload.MISC_CARGO
                        val = self.get_val(Mission.Design.GROSS_MASS)
                        tol = 0.05
                        inputs.set_val(Aircraft.CrewPayload.CARGO_MASS, 0, 'lbm')
                    else:
                        control_var = Aircraft.CrewPayload.WING_CARGO
                        val = wing_cargo
                        tol = 1.1
                else:
                    control_var = Aircraft.CrewPayload.MISC_CARGO
                    val = misc_cargo
                    tol = 1.1

            off_design_prob.model.add_design_var(
                control_var,
                lower=0,
                upper=val * tol,
                ref=val,
            )

        if fill_fuel:
            off_design_prob.model.add_design_var(
                Mission.Summary.GROSS_MASS,
                lower=0,
                upper=off_design_prob.aviary_inputs.get_val(Mission.Design.GROSS_MASS, units='lbm'),
                ref=off_design_prob.aviary_inputs.get_val(Mission.Design.GROSS_MASS, units='lbm'),
            )

        off_design_prob.add_objective(verbosity=verbosity)
        off_design_prob.setup(verbosity=verbosity)
        off_design_prob.set_initial_guesses(verbosity=verbosity)

        off_design_prob.run_aviary_problem(verbosity=verbosity)

        return off_design_prob

    def run_payload_range(self, verbosity=None):
        """
        This function runs Payload/Range analysis for the aircraft model.

        For an aircraft model that has been sized with a mission has has successfully converged,
        this function will adjust the given phase information, assuming that there is a phase named
        'cruise' and elongates the duration bounds to allow the optimizer
        to converge for the max economic and ferry missions.

        Parameters
        ----------
        verbosity : Verbosity or int (optional)
            Sets overriding verbosity to be used while running all payload-range points

        Returns
        -------
        payload_range_problems : tuple
            Tuple containing the off-design AviaryProblems for the max economic and ferry ranges

        TODO currently does not account for reserve fuel
        """
        # For off-design missions, provided verbosity will be used for all L2 method calls
        if verbosity is not None:
            # compatibility with being passed int for verbosity
            verbosity = Verbosity(verbosity)
        else:
            verbosity = self.verbosity  # defaults to BRIEF

        if not self.result.success and verbosity > Verbosity.QUIET:
            warnings.warn(
                'Payload Range Diagrams cannot be generated for unconverged Aviary problems.'
            )
            return ()
        elif self.problem_type is ProblemType.MULTI_MISSION and verbosity > Verbosity.QUIET:
            warnings.warn(
                'Payload Range Diagrams currently cannot be generated for aircraft designed '
                'using multimission capability.'
            )
            return ()

        # Off-design missions do not currently work for GASP masses or missions.
        mass_method = self.model.aviary_inputs.get_val(Settings.MASS_METHOD)
        equations_of_motion = self.model.aviary_inputs.get_val(Settings.EQUATIONS_OF_MOTION)
        if (
            mass_method == LegacyCode.FLOPS
            and equations_of_motion is EquationsOfMotion.HEIGHT_ENERGY
        ):
            # make a copy of the phase_info to avoid modifying the original.
            phase_info = self.model.mission_info.copy()
            phase_info['pre_mission'] = self.model.pre_mission_info.copy()
            phase_info['post_mission'] = self.model.post_mission_info.copy()
            # This checks if the 'cruise' phase exists, then automatically extends duration bounds
            # of the cruise stage to allow for the longer economic and ferry missions.
            if phase_info['cruise']:
                min_duration = phase_info['cruise']['user_options']['time_duration_bounds'][0][0]
                max_duration = phase_info['cruise']['user_options']['time_duration_bounds'][0][1]
                cruise_units = phase_info['cruise']['user_options']['time_duration_bounds'][1]

                phase_info['cruise']['user_options'].update(
                    {'time_duration_bounds': ((min_duration, 2 * max_duration), cruise_units)}
                )

            # TODO Verify that previously run point is actually max payload/fuel point, and if not
            #      run off-design mission for that point
            # Point 1 is along the y axis (range=0)
            payload_1 = float(self.get_val(Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS)[0])
            range_1 = 0
            fuel_1 = 0

            # Point 2 (Design Range): sizing mission which is assumed to be the point of max
            # payload + fuel on the payload and range diagram
            payload_2 = payload_1

            range_2 = float(self.get_val(Mission.Summary.RANGE)[0])
            gross_mass = float(self.get_val(Mission.Summary.GROSS_MASS)[0])
            # NOTE this operating mass is based on the previously run mission - assumed this is the
            # design mission!! Includes cargo containers needed for design (max payload)
            operating_mass = float(self.get_val(Mission.Summary.OPERATING_MASS)[0])
            fuel_capacity = float(self.get_val(Aircraft.Fuel.TOTAL_CAPACITY)[0])
            unusable_fuel = float(self.get_val(Aircraft.Fuel.UNUSABLE_FUEL_MASS)[0])
            max_payload = float(self.get_val(Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS)[0])

            fuel_2 = self.get_val(Mission.Summary.FUEL_BURNED)[0]

            max_usable_fuel = fuel_capacity - unusable_fuel

            # An aircraft may be designed with fuel tank capacity that, if fully filled, would
            # exceed MTOW. In that scenario, Max Economic Range and Ferry Range are the same, and
            # the point only needs to be run once.
            if operating_mass + max_usable_fuel < gross_mass:
                # Point 3 (Max Economic Range): max fuel and remaining payload capacity
                economic_mission_total_payload = gross_mass - operating_mass - max_usable_fuel

                payload_frac = economic_mission_total_payload / max_payload

                # Calculates Different payload quantities
                economic_mission_wing_cargo = (
                    self.model.aviary_inputs.get_val(Aircraft.CrewPayload.WING_CARGO, 'lbm')
                    * payload_frac
                )
                economic_mission_misc_cargo = (
                    self.model.aviary_inputs.get_val(Aircraft.CrewPayload.MISC_CARGO, 'lbm')
                    * payload_frac
                )
                economic_mission_num_first = int(
                    (self.model.aviary_inputs.get_val(Aircraft.CrewPayload.Design.NUM_FIRST_CLASS))
                    * payload_frac
                )
                economic_mission_num_bus = int(
                    (
                        self.model.aviary_inputs.get_val(
                            Aircraft.CrewPayload.Design.NUM_BUSINESS_CLASS
                        )
                    )
                    * payload_frac
                )
                economic_mission_num_tourist = int(
                    (
                        self.model.aviary_inputs.get_val(
                            Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS
                        )
                    )
                    * payload_frac
                )

                # Passenger number rounding and potentially cargo container mass changing means
                # we don't know if we actually filled the aircraft to exactly TOGW yet. Need to use
                # "fill_cargo" flag in off-design call
                economic_range_prob = self.run_off_design_mission(
                    problem_type=ProblemType.FALLOUT,
                    phase_info=phase_info,
                    num_first_class=economic_mission_num_first,
                    num_business=economic_mission_num_bus,
                    num_tourist=economic_mission_num_tourist,
                    wing_cargo=economic_mission_wing_cargo,
                    misc_cargo=economic_mission_misc_cargo,
                    name=self._name + '_max_economic_range',
                    fill_cargo=True,
                    verbosity=verbosity,
                )

                # Pull the payload and range values from the fallout mission
                payload_3 = float(
                    economic_range_prob.get_val(Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS)
                )

                range_3 = float(economic_range_prob.get_val(Mission.Summary.RANGE))
                fuel_3 = economic_range_prob.get_val(Mission.Summary.FUEL_BURNED)[0]

                prob_3_skip = False
            else:
                prob_3_skip = True
                # only fill fuel until hit TOGW
                max_usable_fuel = gross_mass - operating_mass

            # Point 4 (Ferry Range): maximum fuel and 0 payload
            ferry_range_gross_mass = operating_mass + max_usable_fuel
            # BUG 0 passengers breaks the problem, so 1 must be used
            ferry_range_prob = self.run_off_design_mission(
                problem_type=ProblemType.FALLOUT,
                phase_info=phase_info,
                num_first_class=0,
                num_business=0,
                num_tourist=1,
                wing_cargo=0,
                misc_cargo=0,
                cargo_mass=0,
                mission_gross_mass=ferry_range_gross_mass,
                name=self._name + '_ferry_range',
                fill_fuel=True,
                verbosity=verbosity,
            )

            payload_4 = float(ferry_range_prob.get_val(Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS))
            range_4 = float(ferry_range_prob.get_val(Mission.Summary.RANGE))
            fuel_4 = ferry_range_prob.get_val(Mission.Summary.FUEL_BURNED)[0]

            # if economic mission was skipped, economic_range_prob is the same as ferry_range_prob
            if prob_3_skip:
                economic_range_prob = ferry_range_prob
                payload_3 = payload_4
                range_3 = range_4

            # Check if fallout missions ran successfully before writing to csv file
            # If both missions ran successfully, writes the payload/range data to a csv file
            self.payload_range_data = payload_range_data = NamedValues()
            if ferry_range_prob.result.success and economic_range_prob.result.success:
                payload_range_data.set_val(
                    'Mission Name',
                    ['Zero Fuel', 'Design Mission', 'Max Economic Mission', 'Ferry Mission'],
                )
                payload_range_data.set_val(
                    'Payload', [payload_1, payload_2, payload_3, payload_4], 'lbm'
                )
                payload_range_data.set_val('Fuel', [fuel_1, fuel_2, fuel_3, fuel_4], 'lbm')
                payload_range_data.set_val('Range', [range_1, range_2, range_3, range_4], 'NM')

                write_data_file(
                    Path(self.get_reports_dir(force=True)) / 'payload_range_data.csv',
                    payload_range_data,
                )

                # Prints the payload/range data to the console if verbosity is set to VERBOSE or DEBUG
                if verbosity >= Verbosity.VERBOSE:
                    for item in payload_range_data:
                        print(f'{item[0]} ({item[1][1]}): {item[1][0]}')

                return (economic_range_prob, ferry_range_prob)
            else:
                warnings.warn(
                    'One or both of the fallout missions did not run successfully; payload/range '
                    'diagram was not generated.'
                )
        else:
            warnings.warn(
                'Payload/range analysis is currently only supported for the energy equations of '
                'motion.'
            )

    def save_results(self, json_filename='sizing_results.json'):
        """
        This function saves an aviary problem object into a json file.

        Parameters
        ----------
        aviary_problem : AviaryProblem
            Aviary problem object optimized for the aircraft design/sizing mission.
            Assumed to contain aviary_inputs and Mission.Summary.GROSS_MASS
        json_filename : string
            User specified name and relative path of json file to save the data into.
        save_to_reports : bool
            Flag that sets where the results are saved - if True, the file is saved in the OpenMDAO
            reports directory. If False, the file is saved to the current working directory.
        """
        aviary_input_list = []
        with open(json_filename, 'w') as jsonfile:
            # Loop through aviary input datastructure and create a list
            for data in self.model.aviary_inputs:
                (name, (value, units)) = data
                type_value = type(value)

                # Get the gross mass value from the sizing problem and add it to input list
                if name == Mission.Summary.GROSS_MASS or name == Mission.Design.GROSS_MASS:
                    Mission_Summary_GROSS_MASS_val = self.get_val(
                        Mission.Summary.GROSS_MASS, units=units
                    )
                    Mission_Summary_GROSS_MASS_val_list = Mission_Summary_GROSS_MASS_val.tolist()
                    value = Mission_Summary_GROSS_MASS_val_list[0]

                else:
                    # there are different data types we need to handle for conversion to json format
                    # int, bool, float doesn't need anything special

                    # Convert numpy arrays to lists
                    if type_value is np.ndarray:
                        value = value.tolist()

                    # Lists are fine except if they contain enums or Paths
                    if type_value is list:
                        if isinstance(value[0], Enum):
                            for i in range(len(value)):
                                value[i] = value[i].name
                        elif isinstance(value[0], Path):
                            for i in range(len(value)):
                                value[i] = str(value[i])

                    # Enums and Paths need converting to a string
                    if isinstance(value, Enum):
                        value = value.name
                    elif isinstance(value, Path):
                        value = str(value)

                # Append the data to the list
                aviary_input_list.append([name, value, units, str(type_value)])

            if Mission.Design.GROSS_MASS not in self.model.aviary_inputs:
                aviary_input_list.append(
                    [
                        Mission.Design.GROSS_MASS,
                        self.get_val(Mission.Design.GROSS_MASS, 'lbm')[0],
                        'lbm',
                        str(float),
                    ]
                )

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
        takeoff_mass = self.model.aviary_inputs.get_val(Mission.Design.GROSS_MASS, units='lbm')

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

            for name, value_units in sorted(self.model.aviary_inputs):
                value, units = value_units
                writer.writerow({'name': name, 'value': value, 'units': units})


def _read_sizing_json(json_filename, meta_data, verbosity=Verbosity.BRIEF):
    """
    This function reads in saved results from a json file.

    Parameters
    ----------
    json_filename: str, Path
        json file to load the data from
    meta_data: dict
        Variable metadata that will be used to load file data

    Returns
    -------
    AviaryValues object with updated input values from json file
    """
    aviary_inputs = AviaryValues()

    # load saved input list from json file
    with open(json_filename) as json_data_file:
        loaded_aviary_input_list = json.load(json_data_file)
        json_data_file.close()

    # Loop over input list and assign aviary problem input values
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
                            .replace('<', '')
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
                tmp_var_values.replace('>', '')
                .replace('<', '')
                .replace(']', '')
                .replace("'", '')
                .replace(' ', '')
            )
            var_values = convert_strings_to_data([var_values])

        # Check if the variable is in meta data
        if var_name in meta_data.keys():
            try:
                aviary_inputs.set_val(var_name, var_values, units=var_units, meta_data=meta_data)

            except BaseException:
                if verbosity >= Verbosity.VERBOSE:
                    warnings.warn(
                        f'Could not add item in json output to AviaryValues: input string = '
                        f'{inputs}, attempted to set_value({var_name}, {var_values}, {var_units}). '
                        'This variable was skipped.'
                    )
        else:
            # Not in the MetaData
            if verbosity >= Verbosity.VERBOSE:
                warnings.warn(
                    f'While reading json output, item was not found in MetaData: {inputs}. This '
                    'variable was skipped.'
                )

    return aviary_inputs


def reload_aviary_problem(
    filename, phase_info=None, metadata=BaseMetaData.copy(), verbosity=Verbosity.QUIET
):
    """
    Loads a previously sized Aviary model and returns an AviaryProblem for that model.

    Parameters
    ----------
    filename : str, Path
        User specified name and relative path of json file containing the sized aircraft data

    metadata : dict (optional)
        Custom metadata if needed to read all variables present in the json output file

    verbosity : Verbosity, int (optional)
        Controls level of terminal output for function call

    Returns
    -------
    Aviary Problem object with filled aviary_inputs. To use this problem for anything other than
    running off-design missions, then the full level 2 interface should be used. "load_inputs()"
    can be skipped as the "aviary_inputs" attribute is prefilled here.
    """
    # warning if default is used
    # Initialize a new aviary problem and aviary_input data structure
    prob = AviaryProblem()

    filename = get_path(filename)

    aviary_inputs = _read_sizing_json(filename, metadata, verbosity)

    prob.load_inputs(aviary_inputs, phase_info, verbosity=verbosity)

    prob.check_and_preprocess_inputs(verbosity=verbosity)

    # Add Systems
    prob.add_pre_mission_systems(verbosity=verbosity)

    prob.add_phases(verbosity=verbosity)

    prob.add_post_mission_systems(verbosity=verbosity)

    # Link phases and variables
    prob.link_phases(verbosity=verbosity)

    prob.add_driver(verbosity=verbosity)

    prob.add_design_variables(verbosity=verbosity)

    # Load optimization problem formulation
    # Detail which variables the optimizer can control
    prob.add_objective(verbosity=verbosity)

    prob.setup(verbosity=verbosity)

    prob.final_setup()

    # some variables are normally in the problem instead, so add them there too
    prob.set_val(
        Mission.Summary.GROSS_MASS, aviary_inputs.get_val(Mission.Summary.GROSS_MASS, 'lbm'), 'lbm'
    )

    return prob
