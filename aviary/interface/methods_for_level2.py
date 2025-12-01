import csv
import json
import os
import warnings
from datetime import datetime
from enum import Enum
from pathlib import Path
from packaging import version

import dymos as dm
import numpy as np
import openmdao
import openmdao.api as om
from openmdao.utils.om_warnings import warn_deprecation
from openmdao.utils.reports_system import _default_reports
from openmdao.utils.units import convert_units

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

        # When there is only 1 aircraft model/mission, preserve old behavior.
        self.phase_info = self.model.phase_info
        self.aviary_inputs = aviary_inputs
        self.verbosity = verbosity
        if self.problem_type is None:
            # if there are multiple load_inputs() calls, only the problem type from the first aviary_values is used
            self.problem_type = aviary_inputs.get_val(Settings.PROBLEM_TYPE)

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
        # requires get_all_subsystems, which reqiures core_subsystems, which doesn't exist until
        # after check_and_preprocess is assembled
        self._update_metadata_from_subsystems(self.model)  # update meta data with new entries

    def _update_metadata_from_subsystems(self, group):
        """Merge metadata from user-defined subsystems into problem metadata."""

        # loop through phase_info and external subsystems
        for phase_name in group.phase_info:
            external_subsystems = group.get_all_subsystems(
                group.phase_info[phase_name]['external_subsystems']
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

        self.verbosity = sub.verbosity  # TODO: Needs fixed because old verbosity is over-written

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

        A user can override this method with their own pre-mission systems as desired.
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

        A user can override this with their own postmission systems.
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

            if self.problem_type is ProblemType.SIZING:
                self.model.add_objective(Mission.Objectives.FUEL, ref=ref)

            elif self.problem_type is ProblemType.ALTERNATE:
                self.model.add_objective(Mission.Objectives.FUEL, ref=ref)

            elif self.problem_type is ProblemType.FALLOUT:
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

        # Creates a flag to determine if the user would or would not like a payload/range diagram
        payload_range_bool = False
        if self.problem_type is not ProblemType.MULTI_MISSION:
            if Settings.PAYLOAD_RANGE in self.aviary_inputs:
                payload_range_bool = self.aviary_inputs.get_val(Settings.PAYLOAD_RANGE)

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

        # Checks if the payload/range toggle in the aviary inputs csv file has been set and that the
        # current problem is a sizing mission.
        if payload_range_bool:
            self.run_payload_range()

    def run_payload_range(self, verbosity=None):
        """
        This function runs Payload/Range analysis for the aircraft model.

        Assuming that the aircraft model has been sized and the mission has been run and has successfully converged,
        This function will adjust the given phase information by assumming firstly that there is a phase named 'cruise'
        and elongates the duration bounds to allow the optimizer to arrive at a local maximum for the max_fuel_plus_payload and ferry ranges.
        """
        # `self.verbosity` is "true" verbosity for entire run. `verbosity` is verbosity
        # override for just this method
        if verbosity is not None:
            # compatibility with being passed int for verbosity
            verbosity = Verbosity(verbosity)
        else:
            verbosity = self.verbosity  # defaults to BRIEF

        # Checks if the sizing mission has run successfully.
        # If the problem is both a sizing problem has run successfully, if not, we do not run the payload/range function.
        if self.result.success and self.problem_type is ProblemType.SIZING:
            # Off-design missions do not currently work for GASP masses or missions.
            mass_method = self.aviary_inputs.get_val(Settings.MASS_METHOD)
            equations_of_motion = self.aviary_inputs.get_val(Settings.EQUATIONS_OF_MOTION)
            # Checks to determine that both the mass and mission methods are set to FLOPS and Height Energy.
            if (
                mass_method == LegacyCode.FLOPS
                and equations_of_motion is EquationsOfMotion.HEIGHT_ENERGY
            ):
                # Ensure proper transfer of json files.
                self.save_sizing_to_json(json_filename='payload_range_sizing.json')

                # make a copy of the phase_info to avoid modifying the original.
                phase_info = self.model.phase_info
                phase_info['pre_mission'] = self.model.pre_mission_info
                phase_info['post_mission'] = self.model.post_mission_info
                # This checks if the 'cruise' phase exists, then automatically adjusts duration bounds of the cruise stage
                # to allow the optimizer to arrive at a local maxim for the max_fuel_plus_payload and the ferry ranges.
                if phase_info['cruise']:
                    min_duration = phase_info['cruise']['user_options']['time_duration_bounds'][0][
                        0
                    ]
                    max_duration = phase_info['cruise']['user_options']['time_duration_bounds'][0][
                        1
                    ]
                    cruise_units = phase_info['cruise']['user_options']['time_duration_bounds'][1]

                    # Simply doubling the amount of time the optimizer is allowed to stay in the cruise phase, as well as ensure cruise is optimized
                    phase_info['cruise']['user_options'].update(
                        {'time_duration_bounds': ((min_duration, 2 * max_duration), cruise_units)}
                    )

                # point 1 along the y axis (range=0)
                payload_1 = float(self.get_val(Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS)[0])
                range_1 = 0

                # point 2, sizing mission which is assumed to be the point of max payload + fuel on the payload and range diagram
                payload_2 = payload_1
                range_2 = float(self.get_val(Mission.Summary.RANGE)[0])

                # check if fuel capacity does not exceed sizing mission design gross mass
                gross_mass = float(self.get_val(Mission.Summary.GROSS_MASS)[0])
                operating_mass = float(self.get_val(Aircraft.Design.OPERATING_MASS)[0])
                fuel_capacity = float(self.get_val(Aircraft.Fuel.TOTAL_CAPACITY)[0])
                max_payload = float(self.get_val(Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS)[0])

                # When a mission is run with a target range significantly shorter than the aircraft's design range,
                # the "design" mission may not accurately represent the aircraft's sizing requirements. In this scenario,
                # the aircraft would be sized based on gross mass and operating mass values where adding the full fuel
                # capacity to the operating mass would exceed the originally designed gross mass limit.
                #
                # Under these conditions, we will still generate a payload/range diagram, but the max_fuel_plus_payload
                # and max_payload_plus_fuel missions will be identical since the aircraft cannot utilize its full
                # fuel capacity without exceeding design constraints.

                if operating_mass + fuel_capacity < gross_mass:
                    # point 3, fallout mission with max fuel and payload
                    # The payload allowed is the payload that fits on the aircraft at maximum fuel capacity
                    max_fuel_plus_payload_total_payload = (
                        gross_mass - operating_mass - fuel_capacity
                    )

                    payload_frac = max_fuel_plus_payload_total_payload / max_payload

                    # Calculates Different payload quantities
                    max_fuel_plus_payload_wing_cargo = (
                        int(self.aviary_inputs.get_val(Aircraft.CrewPayload.WING_CARGO, 'lbm'))
                        * payload_frac
                    )
                    max_fuel_plus_payload_misc_cargo = (
                        int(self.aviary_inputs.get_val(Aircraft.CrewPayload.MISC_CARGO, 'lbm'))
                        * payload_frac
                    )
                    max_fuel_plus_payload_num_first = int(
                        (self.aviary_inputs.get_val(Aircraft.CrewPayload.Design.NUM_FIRST_CLASS))
                        * payload_frac
                    )
                    max_fuel_plus_payload_num_bus = int(
                        (self.aviary_inputs.get_val(Aircraft.CrewPayload.Design.NUM_BUSINESS_CLASS))
                        * payload_frac
                    )
                    max_fuel_plus_payload_num_tourist = int(
                        (self.aviary_inputs.get_val(Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS))
                        * payload_frac
                    )

                    prob_fallout_max_fuel_plus_payload = self.fallout_mission(
                        json_filename='payload_range_sizing.json',
                        num_first=max_fuel_plus_payload_num_first,
                        num_business=max_fuel_plus_payload_num_bus,
                        num_tourist=max_fuel_plus_payload_num_tourist,
                        wing_cargo=max_fuel_plus_payload_wing_cargo,
                        misc_cargo=max_fuel_plus_payload_misc_cargo,
                        phase_info=phase_info,
                    )

                    # Pull the payload and range values from the fallout mission
                    payload_3 = float(
                        prob_fallout_max_fuel_plus_payload.get_val(
                            Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS
                        )
                    )

                    range_3 = float(
                        prob_fallout_max_fuel_plus_payload.get_val(Mission.Summary.RANGE)
                    )

                    prob_3_skip = False
                else:
                    # If the fuel capacity from the aviary_inputs csv file plus the sized operating mass exceeds the gross mass
                    # the fuel_capacity will be adjusted to equal the difference between the gross mass and the operating mass
                    prob_3_skip = True
                    fuel_capacity = gross_mass - operating_mass

                # Point 4, ferry mission with maximum fuel and 0 payload
                max_fuel_zero_payload_payload = operating_mass + fuel_capacity
                # Aviary does not currently allow for off-design missions of 0 passengers, therefore 1 will be used
                prob_fallout_ferry = self.fallout_mission(
                    json_filename='payload_range_sizing.json',
                    num_first=0,
                    num_business=0,
                    num_tourist=1,
                    num_pax=1,
                    wing_cargo=0,
                    misc_cargo=0,
                    cargo_mass=0,
                    mission_mass=max_fuel_zero_payload_payload,
                    phase_info=phase_info,
                )

                payload_4 = float(
                    prob_fallout_ferry.get_val(Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS)
                )
                range_4 = float(prob_fallout_ferry.get_val(Mission.Summary.RANGE))

                # if problem 3 was skipped, prob_falloout_fuel_plus_payload is redefined so it still exists despite being the same as prob_fallout_ferry
                if prob_3_skip:
                    prob_fallout_max_fuel_plus_payload = prob_fallout_ferry
                    payload_3 = payload_4
                    range_3 = range_4

                # Check if fallout missions ran successfully before writing to csv file
                # If both missions ran successfully, writes the payload/range data to a csv file
                if (
                    prob_fallout_ferry.result.success
                    and prob_fallout_max_fuel_plus_payload.result.success
                ):
                    # TODO Temporary csv writing for payload/range data, should be replaced with a more robust solution
                    csv_filepath = Path(self.get_reports_dir()) / 'payload_range_data.csv'
                    with open(csv_filepath, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)

                        # Write header row
                        writer.writerow(['Point', 'Payload (lbs)', 'Range (NM)'])

                        # Write the four points directly
                        writer.writerow(['Max Payload Zero Fuel', payload_1, range_1])
                        writer.writerow(['Max Payload Plus Fuel', payload_2, range_2])
                        writer.writerow(['Max Fuel Plus Payload', payload_3, range_3])
                        writer.writerow(['Ferry Mission', payload_4, range_4])

                    # Prints the payload/range data to the console if verbosity is set to VERBOSE or DEBUG
                    if verbosity >= Verbosity.VERBOSE:
                        payload_points = [
                            'Payload (lbs)',
                            payload_1,
                            payload_2,
                            payload_3,
                            payload_4,
                        ]
                        range_points = ['Range (NM)', range_1, range_2, range_3, range_4]

                        print(range_points)
                        print(payload_points)

                    return (prob_fallout_max_fuel_plus_payload, prob_fallout_ferry)
                else:
                    warnings.warn(
                        'One or both of the fallout missions did not run successfully; payload/range diagram was not generated.'
                    )
            else:
                warnings.warn(
                    'The payload/range analysis is only supported for FLOPS missions with Height Energy equations of motion; the payload/range analysis will not be run.'
                )
        else:
            if self.problem_type is ProblemType.SIZING:
                warnings.warn(
                    'The sizing problem has not run successfully; therefore, the payload/range analysis will not be run.'
                )
            else:
                warnings.warn('Payload/range analysis is only available for sizing problem types')

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
            # model.phase_info only contains mission information
            phase_info = self.model.phase_info
            phase_info['pre_mission'] = self.model.pre_mission_info
            phase_info['post_mission'] = self.model.post_mission_info
        if mission_range is None:
            # mission range is sliced from a column vector numpy array, i.e. it is a len
            # 1 numpy array
            # fixes issue wherein an alternate mission with no inputs "alternate_mission()" attempts to run the range
            # within the CSV file instead of in the phase_info.
            if mass_method == LegacyCode.GASP:
                mission_range = self.get_val(Mission.Design.RANGE)[0]
            elif mass_method == LegacyCode.FLOPS:
                try:
                    mission_range = self.model.post_mission_info['target_range'][0]
                except:
                    mission_range = self.get_val(Mission.Design.RANGE)[0]

        # gross mass is sliced from a column vector numpy array, i.e. it is a len 1 numpy
        # array
        mission_mass = self.get_val(Mission.Design.GROSS_MASS)[0]
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
        prob_alternate.build_model()
        prob_alternate.add_driver(optimizer, verbosity=verbosity)
        prob_alternate.options = self.options
        prob_alternate.driver.options = self.driver.options
        prob_alternate.driver.opt_settings = self.driver.opt_settings
        prob_alternate.add_design_variables()
        prob_alternate.add_objective()
        prob_alternate.setup()
        if run_mission:
            prob_alternate.run_aviary_problem()
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
            # Somewhere between the sizing and off-design self.pre_mission_info gets deleted
            phase_info = self.model.phase_info
            phase_info['pre_mission'] = self.model.pre_mission_info
            phase_info['post_mission'] = self.model.post_mission_info
        if mission_mass is None:
            # mission mass is sliced from a column vector numpy array, i.e. it is a len 1
            # numpy array
            mission_mass = self.get_val(Mission.Design.GROSS_MASS)[0]
        elif mission_mass > self.get_val(Mission.Design.GROSS_MASS)[0]:
            raise ValueError(
                f'Fallout Mission aircraft gross mass {mission_mass} lbm cannot be greater than Mission.Design.GROSS_MASS {self.get_val(Mission.Design.GROSS_MASS)[0]}'
            )

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
        prob_fallout.build_model()
        prob_fallout.add_driver(optimizer, verbosity=verbosity)
        prob_fallout.options = self.options
        prob_fallout.driver.options = self.driver.options
        prob_fallout.driver.opt_settings = self.driver.opt_settings
        prob_fallout.add_design_variables()
        prob_fallout.add_objective()
        prob_fallout.setup()
        if run_mission:
            prob_fallout.run_aviary_problem()
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
                        type_value = list

                    # Lists are fine except if they contain enums or Paths
                    if type_value == list:
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
            """prob.aviary_inputs.set_val(Mission.Design.RANGE, mission_range, units='NM')"""
            prob.aviary_inputs.set_val(Mission.Summary.RANGE, mission_range, units='NM')
            phase_info['post_mission']['target_range'] = (mission_range, 'nmi')
        # set initial guess for Mission.Summary.GROSS_MASS to help optimizer with new design variable bounds.
        prob.aviary_inputs.set_val(
            Mission.Summary.GROSS_MASS, mission_gross_mass * 0.9, units='lbm'
        )

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
