import csv
import warnings
import inspect
from pathlib import Path
from datetime import datetime
import importlib.util
import sys
import json
import enum

import numpy as np

import dymos as dm
from dymos.utils.misc import _unspecified

import openmdao.api as om
from openmdao.utils.reports_system import _default_reports

from aviary.interface.default_phase_info.two_dof_fiti import add_default_sgm_args
from aviary.interface.utils.check_phase_info import check_phase_info
from aviary.mission.gasp_based.phases.time_integration_traj import FlexibleTraj

from aviary.mission.gasp_based.phases.v_rotate_comp import VRotateComp
from aviary.mission.gasp_based.polynomial_fit import PolynomialFit
from aviary.mission.problem_builder_2DOF import ProblemBuilder2DOF
from aviary.mission.problem_builder_height_energy import ProblemBuilderHeightEnergy
from aviary.mission.problem_builder_solved_2DOF import ProblemBuilderSolved2DOF

from aviary.subsystems.aerodynamics.aerodynamics_builder import CoreAerodynamicsBuilder
from aviary.subsystems.geometry.geometry_builder import CoreGeometryBuilder
from aviary.subsystems.mass.mass_builder import CoreMassBuilder
from aviary.subsystems.premission import CorePreMission
from aviary.subsystems.propulsion.propulsion_builder import CorePropulsionBuilder
from aviary.subsystems.propulsion.utils import build_engine_deck

from aviary.utils.aviary_values import AviaryValues
from aviary.utils.functions import wrapped_convert_units
from aviary.utils.functions import convert_strings_to_data, set_value
from aviary.utils.merge_variable_metadata import merge_meta_data
from aviary.utils.preprocessors import preprocess_options
from aviary.utils.process_input_decks import create_vehicle, update_GASP_options

from aviary.variable_info.enums import AnalysisScheme, ProblemType, EquationsOfMotion, LegacyCode, Verbosity
from aviary.variable_info.functions import setup_trajectory_params, setup_model_options
from aviary.variable_info.variables import Aircraft, Mission, Dynamic, Settings
from aviary.variable_info.variable_meta_data import _MetaData as BaseMetaData

from aviary.core.PostMissionGroup import PostMissionGroup
from aviary.core.PreMissionGroup import PreMissionGroup
from aviary.core.AviaryGroup import AviaryGroup

FLOPS = LegacyCode.FLOPS
GASP = LegacyCode.GASP

TWO_DEGREES_OF_FREEDOM = EquationsOfMotion.TWO_DEGREES_OF_FREEDOM
HEIGHT_ENERGY = EquationsOfMotion.HEIGHT_ENERGY
SOLVED_2DOF = EquationsOfMotion.SOLVED_2DOF
CUSTOM = EquationsOfMotion.CUSTOM


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

    def __init__(self, analysis_scheme=AnalysisScheme.COLLOCATION, **kwargs):
        # Modify OpenMDAO's default_reports for this session.
        new_reports = ['subsystems', 'mission', 'timeseries_csv', 'run_status']
        for report in new_reports:
            if report not in _default_reports:
                _default_reports.append(report)

        super().__init__(**kwargs)

        self.timestamp = datetime.now()

        self.model = AviaryGroup()
        self.pre_mission = PreMissionGroup()
        self.post_mission = PostMissionGroup()

        self.aviary_inputs = None

        self.traj = None

        self.analysis_scheme = analysis_scheme

        self.regular_phases = []
        self.reserve_phases = []
        self.builder = None

    def load_inputs(
            self, aviary_inputs, phase_info=None, engine_builders=None,
            problem_builder=None,
            meta_data=BaseMetaData, verbosity=Verbosity.BRIEF):
        """
        This method loads the aviary_values inputs and options that the
        user specifies. They could specify files to load and values to
        replace here as well.
        Phase info is also loaded if provided by the user. If phase_info is None,
        the appropriate default phase_info based on mission analysis method is used.

        This method is not strictly necessary; a user could also supply
        an AviaryValues object and/or phase_info dict of their own.
        """
        # compatibility with being passed int for verbosity
        verbosity = Verbosity(verbosity)

        ## LOAD INPUT FILE ###
        # Create AviaryValues object from file (or process existing AviaryValues object
        # with default values from metadata) and generate initial guesses
        aviary_inputs, self.initialization_guesses = create_vehicle(
            aviary_inputs, meta_data=meta_data, verbosity=verbosity)

        # pull which methods will be used for subsystems and mission
        self.mission_method = mission_method = aviary_inputs.get_val(
            Settings.EQUATIONS_OF_MOTION)
        self.mass_method = mass_method = aviary_inputs.get_val(Settings.MASS_METHOD)

        # Determine which problem builder to use based on mission_method
        if mission_method is HEIGHT_ENERGY:
            self.builder = ProblemBuilderHeightEnergy()
        elif mission_method is TWO_DEGREES_OF_FREEDOM:
            self.builder = ProblemBuilder2DOF()
        elif mission_method is SOLVED_2DOF:
            self.builder = ProblemBuilderSolved2DOF()
        elif mission_method is CUSTOM:
            if problem_builder:
                self.builder = problem_builder()
                # TODO: make draft / example custom builder
            else:
                raise ValueError(
                    f'When using "settings:equations_of_motion,custom", a problem_builder must be specified in load_inputs().')
        else:
            raise ValueError(
                f'settings:equations_of_motion must be one of: height_energy, 2DOF, or custom')

        # TODO this should be a preprocessor step if it is required here
        if mission_method is TWO_DEGREES_OF_FREEDOM or mass_method is GASP:
            aviary_inputs = update_GASP_options(aviary_inputs)

        ## LOAD PHASE_INFO ###
        if phase_info is None:
            # check if the user generated a phase_info from gui
            # Load the phase info dynamically from the current working directory
            phase_info_module_path = Path.cwd() / 'outputted_phase_info.py'

            if phase_info_module_path.exists():
                spec = importlib.util.spec_from_file_location(
                    'outputted_phase_info', phase_info_module_path)
                outputted_phase_info = importlib.util.module_from_spec(spec)
                sys.modules['outputted_phase_info'] = outputted_phase_info
                spec.loader.exec_module(outputted_phase_info)

                # Access the phase_info variable from the loaded module
                phase_info = outputted_phase_info.phase_info

                # if verbosity level is BRIEF or higher, print that we're using the
                # outputted phase info
                if verbosity is not None and verbosity >= Verbosity.BRIEF:
                    print('Using outputted phase_info from current working directory')
            else:

                phase_info = self.builder.phase_info_default_location(self)

                if verbosity is not None and verbosity >= Verbosity.BRIEF:
                    print('Loaded default phase_info for '
                          f'{self.mission_method.value.lower()} equations of motion')

        # create a new dictionary that only contains the phases from phase_info
        self.phase_info = {}

        for phase_name in phase_info:
            if 'external_subsystems' not in phase_info[phase_name]:
                phase_info[phase_name]['external_subsystems'] = []

            if phase_name not in ['pre_mission', 'post_mission']:
                self.phase_info[phase_name] = phase_info[phase_name]

        # pre_mission and post_mission are stored in their own dictionaries.
        if 'pre_mission' in phase_info:
            self.pre_mission_info = phase_info['pre_mission']
        else:
            self.pre_mission_info = None

        if 'post_mission' in phase_info:
            self.post_mission_info = phase_info['post_mission']
        else:
            self.post_mission_info = None

        if engine_builders is None:
            engine_builders = build_engine_deck(aviary_inputs)
        self.engine_builders = engine_builders

        self.problem_type = aviary_inputs.get_val(Settings.PROBLEM_TYPE)
        self.aviary_inputs = aviary_inputs

        self.builder.initial_guesses(self)
        # This function sets all the following defaults if they were not already set
        # self.engine_builders, self.mass_method, self.pre_mission_info, self_post_mission_info
        # self.require_range_residual, self.target_range
        # other specific self.*** are defined in here as well that are specific to each builder

        return self.aviary_inputs

    def check_and_preprocess_inputs(self):
        """
        This method checks the user-supplied input values for any potential problems
        and preprocesses the inputs to prepare them for use in the Aviary problem.
        """
        aviary_inputs = self.aviary_inputs
        # Target_distance verification for all phases
        # Checks to make sure target_distance is positive,
        for idx, phase_name in enumerate(self.phase_info):
            if 'user_options' in self.phase_info[phase_name]:
                if 'target_distance' in self.phase_info[phase_name]["user_options"]:
                    target_distance = self.phase_info[phase_name]["user_options"][
                        "target_distance"]
                    if target_distance[0] <= 0:
                        raise ValueError(
                            f"Invalid target_distance in [{phase_name}].[user_options]. "
                            f"Current (value: {target_distance[0]}), "
                            f"(units: {target_distance[1]}) <= 0"
                        )

        # Checks to make sure target_duration is positive,
        # Sets duration_bounds, initial_guesses, and fixed_duration
        for idx, phase_name in enumerate(self.phase_info):
            if 'user_options' in self.phase_info[phase_name]:
                analytic = False
                if (self.analysis_scheme is AnalysisScheme.COLLOCATION) and (
                        self.mission_method is EquationsOfMotion.TWO_DEGREES_OF_FREEDOM):
                    try:
                        # if the user provided an option, use it
                        analytic = self.phase_info[phase_name]["user_options"][
                            'analytic']
                    except KeyError:
                        # if it isn't specified, only the default 2DOF cruise for
                        # collocation is analytic
                        if 'cruise' in phase_name:
                            analytic = self.phase_info[phase_name]["user_options"][
                                'analytic'] = True
                        else:
                            analytic = self.phase_info[phase_name]["user_options"][
                                'analytic'] = False

                if 'target_duration' in self.phase_info[phase_name]["user_options"]:
                    target_duration = self.phase_info[phase_name]["user_options"][
                        "target_duration"]
                    if target_duration[0] <= 0:
                        raise ValueError(
                            f'Invalid target_duration in phase_info[{phase_name}]'
                            f'[user_options]. Current (value: {target_duration[0]}), '
                            f'(units: {target_duration[1]}) <= 0")'
                        )

                    # Only applies to non-analytic phases (all HE and most 2DOF)
                    if not analytic:
                        # Set duration_bounds and initial_guesses for time:
                        self.phase_info[phase_name]["user_options"].update({
                            "duration_bounds": ((target_duration[0], target_duration[0]), target_duration[1])})
                        self.phase_info[phase_name].update({"initial_guesses": {"time": (
                            (target_duration[0], target_duration[0]), target_duration[1])}})
                        # Set Fixed_duration to true:
                        self.phase_info[phase_name]["user_options"].update(
                            {"fix_duration": True}
                        )

        if self.analysis_scheme is AnalysisScheme.COLLOCATION:
            check_phase_info(self.phase_info, self.mission_method)

        for phase_name in self.phase_info:
            for external_subsystem in self.phase_info[phase_name]['external_subsystems']:
                aviary_inputs = external_subsystem.preprocess_inputs(
                    aviary_inputs)

        # PREPROCESSORS #
        # Fill in anything missing in the options with computed defaults.
        preprocess_options(aviary_inputs, engine_models=self.engine_builders)

        mission_method = aviary_inputs.get_val(Settings.EQUATIONS_OF_MOTION)
        mass_method = aviary_inputs.get_val(Settings.MASS_METHOD)

        ## Set Up Core Subsystems ##
        if mission_method in (HEIGHT_ENERGY, SOLVED_2DOF):
            everything_else_origin = FLOPS
        elif mission_method is TWO_DEGREES_OF_FREEDOM:
            everything_else_origin = GASP
        else:
            # Custom
            everything_else_origin = None

        prop = CorePropulsionBuilder(
            'core_propulsion', engine_models=self.engine_builders)
        mass = CoreMassBuilder('core_mass', code_origin=self.mass_method)
        aero = CoreAerodynamicsBuilder(
            'core_aerodynamics', code_origin=everything_else_origin)

        # TODO These values are currently hardcoded, in future should come from user
        both_geom = False
        code_origin_to_prioritize = None

        # which geometry methods should be used, or both?
        geom_code_origin = None
        if (everything_else_origin is FLOPS) and (mass_method is FLOPS):
            geom_code_origin = FLOPS
        elif (everything_else_origin is GASP) and (mass_method is GASP):
            geom_code_origin = GASP
        else:
            both_geom = True

        # which geometry method gets prioritized in case of conflicting outputs
        if not code_origin_to_prioritize:
            if everything_else_origin is GASP:
                code_origin_to_prioritize = GASP
            elif everything_else_origin is FLOPS:
                code_origin_to_prioritize = FLOPS

        geom = CoreGeometryBuilder('core_geometry',
                                   code_origin=geom_code_origin,
                                   use_both_geometries=both_geom,
                                   code_origin_to_prioritize=code_origin_to_prioritize)

        subsystems = self.core_subsystems = {'propulsion': prop,
                                             'geometry': geom,
                                             'mass': mass,
                                             'aerodynamics': aero}

        # TODO optionally accept which subsystems to load from phase_info
        default_mission_subsystems = [
            subsystems['aerodynamics'], subsystems['propulsion']]
        self.ode_args = {'aviary_options': aviary_inputs,
                         'core_subsystems': default_mission_subsystems}

        self._update_metadata_from_subsystems()
        self._check_reserve_phase_separation()

    def _update_metadata_from_subsystems(self):
        """
        Merge metadata from user-defined subsystems into problem metadata.
        """
        self.meta_data = BaseMetaData.copy()

        # loop through phase_info and external subsystems
        for phase_name in self.phase_info:

            external_subsystems = self._get_all_subsystems(
                self.phase_info[phase_name]['external_subsystems'])

            for subsystem in external_subsystems:
                meta_data = subsystem.meta_data.copy()
                self.meta_data = merge_meta_data([self.meta_data, meta_data])

    def _check_reserve_phase_separation(self):
        """
        This method checks for reserve=True & False
        Returns an error if a non-reserve phase is specified after a reserve phase.
        return two dictionaries of phases: regular_phases and reserve_phases
        For shooting trajectories, this will also check if a phase is part of the descent
        """

        # Check to ensure no non-reserve phases are specified after reserve phases
        start_reserve = False
        raise_error = False
        for idx, phase_name in enumerate(self.phase_info):
            if 'user_options' in self.phase_info[phase_name]:
                if 'reserve' in self.phase_info[phase_name]["user_options"]:
                    if self.phase_info[phase_name]["user_options"]["reserve"] is False:
                        # This is a regular phase
                        self.regular_phases.append(phase_name)
                        if start_reserve is True:
                            raise_error = True
                    else:
                        # This is a reserve phase
                        self.reserve_phases.append(phase_name)
                        start_reserve = True
                else:
                    # This is a regular phase by default
                    self.regular_phases.append(phase_name)
                    if start_reserve is True:
                        raise_error = True

        if raise_error is True:
            raise ValueError(
                f'In phase_info, reserve=False cannot be specified after a phase where reserve=True. '
                f'All reserve phases must happen after non-reserve phases. '
                f'Regular Phases : {self.regular_phases} | '
                f'Reserve Phases : {self.reserve_phases} ')

        if self.analysis_scheme is AnalysisScheme.SHOOTING:
            self.descent_phases = {}
            for name, info in self.phase_info.items():
                descent = info.get('descent_phase', False)
                if descent:
                    self.descent_phases[name] = info

    def add_pre_mission_systems(self):
        """
        Add pre-mission systems to the Aviary problem. These systems are executed before the mission.

        Depending on the mission model specified (`FLOPS` or `GASP`), this method adds various subsystems
        to the aircraft model. For the `FLOPS` mission model, a takeoff phase is added using the Takeoff class
        with the number of engines and airport altitude specified. For the `GASP` mission model, three subsystems
        are added: a TaxiSegment subsystem, an ExecComp to calculate the time to initiate gear and flaps,
        and an ExecComp to calculate the speed at which to initiate rotation. All subsystems are promoted with
        aircraft and mission inputs and outputs as appropriate.

        A user can override this method with their own pre-mission systems as desired.
        """
        pre_mission = self.pre_mission
        self.model.add_subsystem('pre_mission', pre_mission,
                                 promotes_inputs=['aircraft:*', 'mission:*'],
                                 promotes_outputs=['aircraft:*', 'mission:*'],)

        if 'linear_solver' in self.pre_mission_info:
            pre_mission.linear_solver = self.pre_mission_info['linear_solver']

        if 'nonlinear_solver' in self.pre_mission_info:
            pre_mission.nonlinear_solver = self.pre_mission_info['nonlinear_solver']

        self._add_premission_external_subsystems()

        subsystems = self.core_subsystems

        # Propulsion isn't included in core pre-mission group to avoid override step in
        # configure() - instead add it now
        pre_mission.add_subsystem(
            'core_propulsion', subsystems['propulsion'].build_pre_mission(
                self.aviary_inputs),)

        default_subsystems = [subsystems['geometry'],
                              subsystems['aerodynamics'],
                              subsystems['mass'],]

        pre_mission.add_subsystem(
            'core_subsystems',
            CorePreMission(
                aviary_options=self.aviary_inputs,
                subsystems=default_subsystems,
                process_overrides=False,
            ),
            promotes_inputs=['*'],
            promotes_outputs=['*'])

        if self.pre_mission_info['include_takeoff']:
            self.builder.add_takeoff_systems(self)

    def _add_premission_external_subsystems(self):
        """
        This private method adds each external subsystem to the pre-mission subsystem and
        a mass component that captures external subsystem masses for use in mass buildups.

        Firstly, the method iterates through all external subsystems in the pre-mission
        information. For each subsystem, it builds the pre-mission instance of the
        subsystem.

        Secondly, the method collects the mass names of the added subsystems. This
        expression is then used to define an ExecComp (a component that evaluates a
        simple equation given input values).

        The method promotes the input and output of this ExecComp to the top level of the
        pre-mission object, allowing this calculated subsystem mass to be accessed
        directly from the pre-mission object.
        """

        mass_names = []
        # Loop through all the phases in this subsystem.
        for external_subsystem in self.pre_mission_info['external_subsystems']:
            # Get all the subsystem builders for this phase.
            subsystem_premission = external_subsystem.build_pre_mission(
                self.aviary_inputs)

            if subsystem_premission is not None:
                self.pre_mission.add_subsystem(external_subsystem.name,
                                               subsystem_premission)

                mass_names.extend(external_subsystem.get_mass_names())

        if mass_names:
            formatted_names = []
            for name in mass_names:
                formatted_name = name.replace(':', '_')
                formatted_names.append(formatted_name)

            # Define the expression for computing the sum of masses
            expr = 'subsystem_mass = ' + ' + '.join(formatted_names)

            promotes_inputs_list = [
                (formatted_name, original_name) for formatted_name,
                original_name in zip(formatted_names, mass_names)]

            # Create the ExecComp
            self.pre_mission.add_subsystem(
                'external_comp_sum', om.ExecComp(expr, units='kg'),
                promotes_inputs=promotes_inputs_list,
                promotes_outputs=[('subsystem_mass', Aircraft.Design.
                                   EXTERNAL_SUBSYSTEMS_MASS)])

    def _add_groundroll_eq_constraint(self):
        """
        Add an equality constraint to the problem to ensure that the TAS at the end of the
        groundroll phase is equal to the rotation velocity at the start of the rotation phase.
        """
        self.model.add_subsystem(
            "groundroll_boundary",
            om.EQConstraintComp(
                "velocity",
                eq_units="ft/s",
                normalize=True,
                add_constraint=True,
            ),
        )
        self.model.connect(Mission.Takeoff.ROTATION_VELOCITY,
                           "groundroll_boundary.rhs:velocity")
        self.model.connect(
            "traj.groundroll.states:velocity",
            "groundroll_boundary.lhs:velocity",
            src_indices=[-1],
            flat_src_indices=True,
        )

    def _get_phase(self, phase_name, phase_idx):
        base_phase_options = self.phase_info[phase_name]

        # We need to exclude some things from the phase_options that we pass down
        # to the phases. Intead of "popping" keys, we just create new outer dictionaries.

        phase_options = {}
        for key, val in base_phase_options.items():
            phase_options[key] = val

        phase_options['user_options'] = {}
        for key, val in base_phase_options['user_options'].items():
            phase_options['user_options'][key] = val

        # TODO optionally accept which subsystems to load from phase_info
        subsystems = self.core_subsystems
        default_mission_subsystems = [
            subsystems['aerodynamics'], subsystems['propulsion']]

        phase_builder = self.builder.get_phase_builder(self, phase_name, phase_options)

        phase_object = phase_builder.from_phase_info(
            phase_name, phase_options, default_mission_subsystems,
            meta_data=self.meta_data)

        phase = phase_object.build_phase(aviary_options=self.aviary_inputs)

        self.phase_objects.append(phase_object)

        # TODO: add logic to filter which phases get which controls.
        # right now all phases get all controls added from every subsystem.
        # for example, we might only want ELECTRIC_SHAFT_POWER applied during the
        # climb phase.
        all_subsystems = self._get_all_subsystems(
            phase_options['external_subsystems'])

        # loop through all_subsystems and call `get_controls` on each subsystem
        for subsystem in all_subsystems:
            # add the controls from the subsystems to each phase
            arg_spec = inspect.getfullargspec(subsystem.get_controls)
            if 'phase_name' in arg_spec.args:
                control_dicts = subsystem.get_controls(
                    phase_name=phase_name)
            else:
                control_dicts = subsystem.get_controls(
                    phase_name=phase_name)
            for control_name, control_dict in control_dicts.items():
                phase.add_control(control_name, **control_dict)

        user_options = AviaryValues(phase_options.get('user_options', ()))

        self.builder.set_phase_options(self, phase_name, phase_idx, phase, user_options)

        return phase

    def add_phases(self, phase_info_parameterization=None):
        """
        Add the mission phases to the problem trajectory based on the user-specified
        phase_info dictionary.

        Parameters
        ----------
        phase_info_parameterization (function, optional): A function that takes in the phase_info dictionary
            and aviary_inputs and returns modified phase_info. Defaults to None.

        Returns
        -------
        traj: The Dymos Trajectory object containing the added mission phases.
        """
        if phase_info_parameterization is not None:
            self.phase_info, self.post_mission_info = phase_info_parameterization(
                self.phase_info, self.post_mission_info, self.aviary_inputs)

        phase_info = self.phase_info

        if self.analysis_scheme is AnalysisScheme.COLLOCATION:
            phases = list(phase_info.keys())
            traj = self.model.add_subsystem('traj', dm.Trajectory())

        elif self.analysis_scheme is AnalysisScheme.SHOOTING:
            vb = self.aviary_inputs.get_val(Settings.VERBOSITY)
            add_default_sgm_args(self.phase_info, self.ode_args, vb)

            full_traj = FlexibleTraj(
                Phases=self.phase_info,
                traj_final_state_output=[
                    Dynamic.Vehicle.MASS,
                    Dynamic.Mission.DISTANCE,
                ],
                traj_initial_state_input=[
                    Dynamic.Vehicle.MASS,
                    Dynamic.Mission.DISTANCE,
                    Dynamic.Mission.ALTITUDE,
                ],
                traj_event_trigger_input=[
                    # specify ODE, output_name, with units that SimuPyProblem expects
                    # assume event function is of form ODE.output_name - value
                    # third key is event_idx associated with input
                    (
                        'groundroll',
                        Dynamic.Mission.VELOCITY,
                        0,
                    ),
                    (
                        'climb3',
                        Dynamic.Mission.ALTITUDE,
                        0,
                    ),
                    (
                        'cruise',
                        Dynamic.Vehicle.MASS,
                        0,
                    ),
                ],
                traj_intermediate_state_output=[
                    ('cruise', Dynamic.Mission.DISTANCE),
                    ('cruise', Dynamic.Vehicle.MASS),
                ],
            )
            traj = self.model.add_subsystem(
                'traj', full_traj, promotes_inputs=[
                    ('altitude_initial', Mission.Design.CRUISE_ALTITUDE)])

            self.model.add_subsystem(
                'actual_descent_fuel', om.ExecComp(
                    'actual_descent_fuel = traj_cruise_mass_final - traj_mass_final',
                    actual_descent_fuel={'units': 'lbm'},
                    traj_cruise_mass_final={'units': 'lbm'},
                    traj_mass_final={'units': 'lbm'},))

            self.model.connect('start_of_descent_mass', 'traj.SGMCruise_mass_trigger')
            self.model.connect(
                'traj.mass_final',
                'actual_descent_fuel.traj_mass_final',
                src_indices=[-1],
                flat_src_indices=True,
            )
            self.model.connect(
                'traj.cruise_mass_final',
                'actual_descent_fuel.traj_cruise_mass_final',
                src_indices=[-1],
                flat_src_indices=True,
            )
            self.traj = full_traj
            return traj

        def add_subsystem_timeseries_outputs(phase, phase_name):
            phase_options = self.phase_info[phase_name]
            all_subsystems = self._get_all_subsystems(
                phase_options['external_subsystems'])
            for subsystem in all_subsystems:
                timeseries_to_add = subsystem.get_outputs()
                for timeseries in timeseries_to_add:
                    phase.add_timeseries_output(timeseries)

        if self.analysis_scheme is AnalysisScheme.COLLOCATION:
            self.phase_objects = []
            for phase_idx, phase_name in enumerate(phases):
                phase = traj.add_phase(
                    phase_name, self._get_phase(phase_name, phase_idx))
                add_subsystem_timeseries_outputs(phase, phase_name)

                if self.mission_method is TWO_DEGREES_OF_FREEDOM:

                    # In GASP, we still use the phase name to infer the phase type.
                    # We need this information to be available in the builders.
                    # TODO - Ultimately we should overhaul all of this.
                    self.phase_info[phase_name]['phase_type'] = phase_name

                    if phase_name == 'ascent':
                        self._add_groundroll_eq_constraint()

        # loop through phase_info and external subsystems
        external_parameters = {}
        for phase_name in self.phase_info:
            external_parameters[phase_name] = {}
            all_subsystems = self._get_all_subsystems(
                self.phase_info[phase_name]['external_subsystems'])
            for subsystem in all_subsystems:
                parameter_dict = subsystem.get_parameters(
                    phase_info=self.phase_info[phase_name],
                    aviary_inputs=self.aviary_inputs
                )
                for parameter in parameter_dict:
                    external_parameters[phase_name][parameter] = parameter_dict[parameter]

        traj = setup_trajectory_params(
            self.model, traj, self.aviary_inputs, phases, meta_data=self.meta_data,
            external_parameters=external_parameters)

        if self.mission_method is HEIGHT_ENERGY:
            if not self.pre_mission_info['include_takeoff']:
                first_flight_phase_name = list(phase_info.keys())[0]
                first_flight_phase = traj._phases[first_flight_phase_name]
                first_flight_phase.set_state_options(
                    Dynamic.Vehicle.MASS, fix_initial=False
                )

        self.traj = traj

        return traj

    def add_post_mission_systems(self, include_landing=True):
        """
        Add post-mission systems to the aircraft model. This is akin to the pre-mission group
        or the "premission_systems", but occurs after the mission in the execution order.

        Depending on the mission model specified (`FLOPS` or `GASP`), this method adds various subsystems
        to the aircraft model. For the `FLOPS` mission model, a landing phase is added using the Landing class
        with the wing area and lift coefficient specified, and a takeoff constraints ExecComp is added to enforce
        mass, range, velocity, and altitude continuity between the takeoff and climb phases. The landing subsystem
        is promoted with aircraft and mission inputs and outputs as appropriate, while the takeoff constraints ExecComp
        is only promoted with mission inputs and outputs.

        For the `GASP` mission model, four subsystems are added: a LandingSegment subsystem, an ExecComp to calculate
        the reserve fuel required, an ExecComp to calculate the overall fuel burn, and three ExecComps to calculate
        various mission objectives and constraints. All subsystems are promoted with aircraft and mission inputs and
        outputs as appropriate.

        A user can override this with their own postmission systems.
        """

        if self.pre_mission_info['include_takeoff']:
            self.builder.add_post_mission_takeoff_systems(self)

        if include_landing and self.post_mission_info['include_landing']:
            self.builder.add_landing_systems(self)

        self.model.add_subsystem('post_mission', self.post_mission,
                                 promotes_inputs=['*'],
                                 promotes_outputs=['*'])

        # Add all post-mission external subsystems.
        for external_subsystem in self.post_mission_info['external_subsystems']:
            subsystem_postmission = external_subsystem.build_post_mission(
                self.aviary_inputs)

            if subsystem_postmission is not None:
                self.post_mission.add_subsystem(external_subsystem.name,
                                                subsystem_postmission)

        # Check if regular_phases[] is accessible
        try:
            self.regular_phases[0]
        except BaseException:
            raise ValueError(
                f"regular_phases[] dictionary is not accessible."
                f" For HEIGHT_ENERGY and SOLVED_2DOF missions, check_and_preprocess_inputs()"
                f" must be called before add_post_mission_systems().")

        # Fuel burn in regular phases
        ecomp = om.ExecComp('fuel_burned = initial_mass - mass_final',
                            initial_mass={'units': 'lbm'},
                            mass_final={'units': 'lbm'},
                            fuel_burned={'units': 'lbm'})

        self.post_mission.add_subsystem(
            'fuel_burned', ecomp,
            promotes=[('fuel_burned', Mission.Summary.FUEL_BURNED)])

        if self.analysis_scheme is AnalysisScheme.SHOOTING:
            # shooting method currently doesn't have timeseries
            self.post_mission.promotes('fuel_burned', [
                ('initial_mass', Mission.Summary.GROSS_MASS),
                ('mass_final', Mission.Landing.TOUCHDOWN_MASS),
            ])
        else:
            if self.pre_mission_info['include_takeoff']:
                self.post_mission.promotes('fuel_burned', [
                    ('initial_mass', Mission.Summary.GROSS_MASS),
                ])
            else:
                # timeseries has to be used because Breguet cruise phases don't have
                # states
                self.model.connect(f"traj.{self.regular_phases[0]}.timeseries.mass",
                                   "fuel_burned.initial_mass", src_indices=[0])

            self.model.connect(f"traj.{self.regular_phases[-1]}.timeseries.mass",
                               "fuel_burned.mass_final", src_indices=[-1])

        # Fuel burn in reserve phases
        if self.reserve_phases:
            ecomp = om.ExecComp('reserve_fuel_burned = initial_mass - mass_final',
                                initial_mass={'units': 'lbm'},
                                mass_final={'units': 'lbm'},
                                reserve_fuel_burned={'units': 'lbm'})

            self.post_mission.add_subsystem(
                'reserve_fuel_burned', ecomp, promotes=[
                    ('reserve_fuel_burned', Mission.Summary.RESERVE_FUEL_BURNED)])

            if self.analysis_scheme is AnalysisScheme.SHOOTING:
                # shooting method currently doesn't have timeseries
                self.post_mission.promotes('reserve_fuel_burned', [
                    ('initial_mass', Mission.Landing.TOUCHDOWN_MASS),
                ])
                self.model.connect(
                    f"traj.{self.reserve_phases[-1]}.states:mass",
                    "reserve_fuel_burned.mass_final", src_indices=[-1])
            else:
                # timeseries has to be used because Breguet cruise phases don't have
                # states
                self.model.connect(
                    f"traj.{self.reserve_phases[0]}.timeseries.mass",
                    "reserve_fuel_burned.initial_mass", src_indices=[0])
                self.model.connect(
                    f"traj.{self.reserve_phases[-1]}.timeseries.mass",
                    "reserve_fuel_burned.mass_final", src_indices=[-1])

        self._add_fuel_reserve_component()

        # TODO: need to add some sort of check that this value is less than the fuel capacity
        # TODO: the overall_fuel variable is the burned fuel plus the reserve, but should
        # also include the unused fuel, and the hierarchy variable name should be
        # more clear
        ecomp = om.ExecComp('overall_fuel = (1 + fuel_margin/100)*fuel_burned + reserve_fuel',
                            overall_fuel={'units': 'lbm', 'shape': 1},
                            fuel_margin={"units": "unitless", 'val': 0},
                            fuel_burned={'units': 'lbm'},  # from regular_phases only
                            reserve_fuel={'units': 'lbm', 'shape': 1},
                            )
        self.post_mission.add_subsystem(
            'fuel_calc', ecomp,
            promotes_inputs=[
                ("fuel_margin", Aircraft.Fuel.FUEL_MARGIN),
                ('fuel_burned', Mission.Summary.FUEL_BURNED),
                ("reserve_fuel", Mission.Design.RESERVE_FUEL),
            ],
            promotes_outputs=[('overall_fuel', Mission.Summary.TOTAL_FUEL_MASS)])

        # If a target distance (or time) has been specified for this phase
        # distance (or time) is measured from the start of this phase to the end
        # of this phase
        for phase_name in self.phase_info:
            if 'target_distance' in self.phase_info[phase_name]["user_options"]:
                target_distance = wrapped_convert_units(
                    self.phase_info[phase_name]["user_options"]
                    ["target_distance"],
                    'nmi')
                self.post_mission.add_subsystem(
                    f"{phase_name}_distance_constraint", om.ExecComp(
                        "distance_resid = target_distance - (final_distance - initial_distance)",
                        distance_resid={'units': 'nmi'},
                        target_distance={'val': target_distance, 'units': 'nmi'},
                        final_distance={'units': 'nmi'},
                        initial_distance={'units': 'nmi'},))
                self.model.connect(
                    f"traj.{phase_name}.timeseries.distance",
                    f"{phase_name}_distance_constraint.final_distance",
                    src_indices=[-1])
                self.model.connect(
                    f"traj.{phase_name}.timeseries.distance",
                    f"{phase_name}_distance_constraint.initial_distance",
                    src_indices=[0])
                self.model.add_constraint(
                    f"{phase_name}_distance_constraint.distance_resid", equals=0.0, ref=1e2)

            # this is only used for analytic phases with a target duration
            if 'target_duration' in self.phase_info[phase_name]["user_options"] and \
                    self.phase_info[phase_name]["user_options"].get("analytic", False):
                target_duration = wrapped_convert_units(
                    self.phase_info[phase_name]["user_options"]
                    ["target_duration"],
                    'min')
                self.post_mission.add_subsystem(
                    f"{phase_name}_duration_constraint", om.ExecComp(
                        "duration_resid = target_duration - (final_time - initial_time)",
                        duration_resid={'units': 'min'},
                        target_duration={'val': target_duration, 'units': 'min'},
                        final_time={'units': 'min'},
                        initial_time={'units': 'min'},))
                self.model.connect(
                    f"traj.{phase_name}.timeseries.time",
                    f"{phase_name}_duration_constraint.final_time", src_indices=[-1])
                self.model.connect(
                    f"traj.{phase_name}.timeseries.time",
                    f"{phase_name}_duration_constraint.initial_time",
                    src_indices=[0])
                self.model.add_constraint(
                    f"{phase_name}_duration_constraint.duration_resid", equals=0.0, ref=1e2)

        if self.mission_method in (TWO_DEGREES_OF_FREEDOM, HEIGHT_ENERGY):
            self._add_comps_for_objs_and_cons()

        ecomp = om.ExecComp(
            'mass_resid = operating_empty_mass + overall_fuel + payload_mass -'
            ' initial_mass',
            operating_empty_mass={'units': 'lbm'},
            overall_fuel={'units': 'lbm'},
            payload_mass={'units': 'lbm'},
            initial_mass={'units': 'lbm'},
            mass_resid={'units': 'lbm'})

        if self.mass_method is GASP:
            payload_mass_src = Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS
        else:
            payload_mass_src = Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS

        self.post_mission.add_subsystem(
            'mass_constraint', ecomp,
            promotes_inputs=[
                ('operating_empty_mass', Aircraft.Design.OPERATING_MASS),
                ('overall_fuel', Mission.Summary.TOTAL_FUEL_MASS),
                ('payload_mass', payload_mass_src),
                ('initial_mass', Mission.Summary.GROSS_MASS)],
            promotes_outputs=[("mass_resid", Mission.Constraints.MASS_RESIDUAL)])

        if self.mission_method in (HEIGHT_ENERGY, TWO_DEGREES_OF_FREEDOM):
            self.post_mission.add_constraint(
                Mission.Constraints.MASS_RESIDUAL, equals=0.0, ref=1.e5)

        if self.mission_method is HEIGHT_ENERGY:
            # connect summary mass to the initial guess of mass in the first phase
            if not self.pre_mission_info['include_takeoff']:
                first_flight_phase_name = list(self.phase_info.keys())[0]
                eq = self.model.add_subsystem(
                    f'link_{first_flight_phase_name}_mass', om.EQConstraintComp(),
                    promotes_inputs=[('rhs:mass', Mission.Summary.GROSS_MASS)])
                eq.add_eq_output('mass', eq_units='lbm', normalize=False,
                                 ref=100000., add_constraint=True)
                self.model.connect(
                    f'traj.{first_flight_phase_name}.states:mass',
                    f'link_{first_flight_phase_name}_mass.lhs:mass',
                    src_indices=[0],
                    flat_src_indices=True,
                )

    def _link_phases_helper_with_options(self, phases, option_name, var, **kwargs):
        # Initialize a list to keep track of indices where option_name is True
        true_option_indices = []

        # Loop through phases to find where option_name is True
        for idx, phase_name in enumerate(phases):
            if self.phase_info[phase_name]['user_options'].get(option_name, False):
                true_option_indices.append(idx)

        # Determine the groups of phases to link based on consecutive indices
        groups_to_link = []
        current_group = []

        for idx in true_option_indices:
            if not current_group or idx == current_group[-1] + 1:
                # If the current index is consecutive, add it to the current group
                current_group.append(idx)
            else:
                # Otherwise, start a new group and save the previous one
                groups_to_link.append(current_group)
                current_group = [idx]

        # Add the last group if it exists
        if current_group:
            groups_to_link.append(current_group)

        # Loop through each group and determine the phases to link
        for group in groups_to_link:
            # Extend the group to include the phase before the first True option and
            # after the last True option, if applicable
            if group[0] > 0:
                group.insert(0, group[0] - 1)
            if group[-1] < len(phases) - 1:
                group.append(group[-1] + 1)

            # Extract the phase names for the current group
            phases_to_link = [phases[idx] for idx in group]

            # Link the phases for the current group
            if len(phases_to_link) > 1:
                self.traj.link_phases(phases=phases_to_link, vars=[var], **kwargs)

    def link_phases(self):
        """
        Link phases together after they've been added.

        Based on which phases the user has selected, we might need
        special logic to do the Dymos linkages correctly. Some of those
        connections for the simple GASP and FLOPS mission are shown here.
        """
        self._add_bus_variables_and_connect()

        phases = list(self.phase_info.keys())

        if len(phases) <= 1:
            return

        # In summary, the following code loops over all phases in self.phase_info, gets
        # the linked variables from each external subsystem in each phase, and stores
        # the lists of linked variables in lists_to_link. It then gets a list of
        # unique variable names from lists_to_link and loops over them, creating
        # a list of phase names for each variable and linking the phases
        # using self.traj.link_phases().

        lists_to_link = []
        for idx, phase_name in enumerate(self.phase_info):
            lists_to_link.append([])
            for external_subsystem in self.phase_info[phase_name]['external_subsystems']:
                lists_to_link[idx].extend(external_subsystem.get_linked_variables())

        # get unique variable names from lists_to_link
        unique_vars = list(set([var for sublist in lists_to_link for var in sublist]))

        # Phase linking.
        # If we are under mpi, and traj.phases is running in parallel, then let the
        # optimizer handle the linkage constraints.  Note that we can technically
        # paralellize connected phases, but it requires a solver that we would like
        # to avoid.
        true_unless_mpi = True
        if self.comm.size > 1 and self.traj.options['parallel_phases']:
            true_unless_mpi = False

        # loop over unique variable names
        for var in unique_vars:
            phases_to_link = []
            for idx, phase_name in enumerate(self.phase_info):
                if var in lists_to_link[idx]:
                    phases_to_link.append(phase_name)

            if len(phases_to_link) > 1:  # TODO: hack
                self.traj.link_phases(phases=phases_to_link, vars=[var], connected=True)

        self.builder.link_phases(self, phases, direct_links=true_unless_mpi)

    def add_driver(
            self, optimizer=None, use_coloring=None, max_iter=50,
            verbosity=Verbosity.BRIEF):
        """
        Add an optimization driver to the Aviary problem.

        Depending on the provided optimizer, the method instantiates the relevant driver (ScipyOptimizeDriver or
        pyOptSparseDriver) and sets the optimizer options. Options for 'SNOPT', 'IPOPT', and 'SLSQP' are
        specified. The method also allows for the declaration of coloring and setting debug print options.

        Parameters
        ----------
        optimizer : str
            The name of the optimizer to use. It can be "SLSQP", "SNOPT", "IPOPT" or others supported by OpenMDAO.
            If "SLSQP", it will instantiate a ScipyOptimizeDriver, else it will instantiate a pyOptSparseDriver.

        use_coloring : bool, optional
            If True (default), the driver will declare coloring, which can speed up derivative computations.

        max_iter : int, optional
            The maximum number of iterations allowed for the optimization process. Default is 50. This option is
            applicable to "SNOPT", "IPOPT", and "SLSQP" optimizers.

        verbosity : Verbosity, int or list, optional
            If Verbosity.DEBUG, debug print options ['desvars','ln_cons','nl_cons','objs'] will be set. If a list is
            provided, it will be used as the debug print options.

        Returns
        -------
        None
        """
        # compatibility with being passed int for verbosity
        verbosity = Verbosity(verbosity)

        # Set defaults for optimizer and use_coloring based on analysis scheme
        if optimizer is None:
            optimizer = 'IPOPT' if self.analysis_scheme is AnalysisScheme.SHOOTING else 'SNOPT'
        if use_coloring is None:
            use_coloring = False if self.analysis_scheme is AnalysisScheme.SHOOTING else True

        # check if optimizer is SLSQP
        if optimizer == "SLSQP":
            driver = self.driver = om.ScipyOptimizeDriver()
        else:
            driver = self.driver = om.pyOptSparseDriver()

        driver.options["optimizer"] = optimizer
        if use_coloring:
            driver.declare_coloring()

        if driver.options["optimizer"] == "SNOPT":
            if verbosity == Verbosity.QUIET:
                isumm, iprint = 0, 0
            elif verbosity == Verbosity.BRIEF:
                isumm, iprint = 6, 0
            elif verbosity > Verbosity.BRIEF:
                isumm, iprint = 6, 9
            driver.opt_settings["Major iterations limit"] = max_iter
            driver.opt_settings["Major optimality tolerance"] = 1e-4
            driver.opt_settings["Major feasibility tolerance"] = 1e-7
            driver.opt_settings["iSumm"] = isumm
            driver.opt_settings["iPrint"] = iprint

        elif driver.options["optimizer"] == "IPOPT":
            if verbosity == Verbosity.QUIET:
                print_level = 3  # minimum to get exit status
                driver.opt_settings['print_user_options'] = 'no'
            elif verbosity == Verbosity.BRIEF:
                print_level = 5
                driver.opt_settings['print_user_options'] = 'no'
                driver.opt_settings['print_frequency_iter'] = 10
            elif verbosity == Verbosity.VERBOSE:
                print_level = 5
            else:  # DEBUG
                print_level = 7
            driver.opt_settings['tol'] = 1.0E-6
            driver.opt_settings['mu_init'] = 1e-5
            driver.opt_settings['max_iter'] = max_iter
            driver.opt_settings['print_level'] = print_level
            # for faster convergence
            driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
            driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'
            driver.opt_settings['mu_strategy'] = 'monotone'

        elif driver.options["optimizer"] == "SLSQP":
            if verbosity == Verbosity.QUIET:
                disp = False
            else:
                disp = True
            driver.options["tol"] = 1e-9
            driver.options["maxiter"] = max_iter
            driver.options["disp"] = disp

        if verbosity > Verbosity.QUIET:
            if isinstance(verbosity, list):
                driver.options['debug_print'] = verbosity
            elif verbosity == Verbosity.DEBUG:
                driver.options['debug_print'] = ['desvars', 'ln_cons', 'nl_cons', 'objs']
        if optimizer in ("SNOPT", "IPOPT"):
            if verbosity == Verbosity.QUIET:
                driver.options['print_results'] = False
            elif verbosity < Verbosity.DEBUG:
                driver.options['print_results'] = 'minimal'

    def add_design_variables(self):
        """
        Adds design variables to the Aviary problem.

        Depending on the mission model and problem type, different design variables and constraints
        are added.

        If using the FLOPS model, a design variable is added for the gross mass of the aircraft,
        with a lower bound of 10 lbm and an upper bound of 900,000 lbm.

        If using the GASP model, the following design variables are added depending on the mission type:
            - the initial thrust-to-weight ratio of the aircraft during ascent
            - the duration of the ascent phase
            - the time constant for the landing gear actuation
            - the time constant for the flaps actuation

        In addition, two constraints are added for the GASP model:
            - the initial altitude of the aircraft with gear extended is constrained to be 50 ft
            - the initial altitude of the aircraft with flaps extended is constrained to be 400 ft

        If solving a sizing problem, a design variable is added for the gross mass of the aircraft,
        and another for the gross mass of the aircraft computed during the mission. A constraint
        is also added to ensure that the residual range is zero.

        If solving an alternate problem, only a design variable for the gross mass of the aircraft
        computed during the mission is added. A constraint is also added to ensure that the residual
        range is zero.

        In all cases, a design variable is added for the final cruise mass of the aircraft, with
        no upper bound, and a residual mass constraint is added to ensure that the mass balances.

        """
        # add the engine builder `get_design_vars` dict to a collected dict from
        # the external subsystems

        # TODO : maybe in the most general case we need to handle DVs in the mission and post-mission as well.
        # for right now we just handle pre_mission
        all_subsystems = self._get_all_subsystems()

        # loop through all_subsystems and call `get_design_vars` on each subsystem
        for subsystem in all_subsystems:
            dv_dict = subsystem.get_design_vars()
            for dv_name, dv_dict in dv_dict.items():
                self.model.add_design_var(dv_name, **dv_dict)

        if self.mission_method is SOLVED_2DOF:
            optimize_mass = self.pre_mission_info.get('optimize_mass')
            if optimize_mass:
                self.model.add_design_var(Mission.Design.GROSS_MASS, units='lbm',
                                          lower=10, upper=900.e3, ref=175.e3)

        elif self.mission_method in (HEIGHT_ENERGY, TWO_DEGREES_OF_FREEDOM):
            # vehicle sizing problem
            # size the vehicle (via design GTOW) to meet a target range using all fuel
            # capacity
            if self.problem_type is ProblemType.SIZING:
                self.model.add_design_var(
                    Mission.Design.GROSS_MASS,
                    lower=10.0,
                    upper=None,
                    units='lbm',
                    ref=175e3,
                )
                self.model.add_design_var(
                    Mission.Summary.GROSS_MASS,
                    lower=10.0,
                    upper=None,
                    units='lbm',
                    ref=175e3,
                )

                self.model.add_subsystem(
                    'gtow_constraint',
                    om.EQConstraintComp(
                        'GTOW',
                        eq_units='lbm',
                        normalize=True,
                        add_constraint=True,
                    ),
                    promotes_inputs=[
                        ('lhs:GTOW', Mission.Design.GROSS_MASS),
                        ('rhs:GTOW', Mission.Summary.GROSS_MASS),
                    ],
                )

                if self.require_range_residual:
                    self.model.add_constraint(
                        Mission.Constraints.RANGE_RESIDUAL, equals=0, ref=10
                    )

            # target range problem
            # fixed vehicle (design GTOW) but variable actual GTOW for off-design
            # mission range
            elif self.problem_type is ProblemType.ALTERNATE:
                self.model.add_design_var(
                    Mission.Summary.GROSS_MASS,
                    lower=10.,
                    upper=900e3,
                    units='lbm',
                    ref=175e3,
                )

                self.model.add_constraint(
                    Mission.Constraints.RANGE_RESIDUAL, equals=0, ref=10
                )

            elif self.problem_type is ProblemType.FALLOUT:
                print('No design variables for Fallout missions')

            elif self.problem_type is ProblemType.MULTI_MISSION:
                self.model.add_design_var(
                    Mission.Summary.GROSS_MASS,
                    lower=10.,
                    upper=900e3,
                    units='lbm',
                    ref=175e3,
                )

                self.model.add_constraint(
                    Mission.Constraints.RANGE_RESIDUAL, equals=0, ref=10
                )

                # We must ensure that design.gross_mass is greater than mission.summary.gross_mass
                # and this must hold true for each of the different missions that is flown
                # the result will be the design.gross_mass should be equal to the mission.summary.gross_mass
                # of the heaviest mission
                self.model.add_subsystem(
                    "GROSS_MASS_constraint",
                    om.ExecComp(
                        "gross_mass_resid = design_mass - actual_mass",
                        design_mass={"val": 1, "units": "kg"},
                        actual_mass={"val": 0, "units": "kg"},
                        gross_mass_resid={"val": 30, "units": "kg"},
                    ),
                    promotes_inputs=[
                        ("design_mass", Mission.Design.GROSS_MASS),
                        ("actual_mass", Mission.Summary.GROSS_MASS),
                    ],
                    promotes_outputs=["gross_mass_resid"],
                )

                self.model.add_constraint(
                    "gross_mass_resid", lower=0
                )

            if self.mission_method is TWO_DEGREES_OF_FREEDOM and self.analysis_scheme is AnalysisScheme.COLLOCATION:
                # problem formulation to make the trajectory work
                self.model.add_design_var(Mission.Takeoff.ASCENT_T_INTIIAL,
                                          lower=0, upper=100, ref=30.0)
                self.model.add_design_var(Mission.Takeoff.ASCENT_DURATION,
                                          lower=1, upper=1000, ref=10.)
                self.model.add_design_var("tau_gear", lower=0.01,
                                          upper=1.0, units="unitless", ref=1)
                self.model.add_design_var("tau_flaps", lower=0.01,
                                          upper=1.0, units="unitless", ref=1)
                self.model.add_constraint(
                    "h_fit.h_init_gear", equals=50.0, units="ft", ref=50.0)
                self.model.add_constraint("h_fit.h_init_flaps",
                                          equals=400.0, units="ft", ref=400.0)

    def add_objective(self, objective_type=None, ref=None):
        """
        Add the objective function based on the given objective_type and ref.

        NOTE: the ref value should be positive for values you're trying
        to minimize and negative for values you're trying to maximize.
        Please check and double-check that your ref value makes sense
        for the objective you're using.

        Parameters
        ----------
        objective_type : str
            The type of objective to add. Options are 'mass', 'hybrid_objective', 'fuel_burned', and 'fuel'.
        ref : float
            The reference value for the objective. If None, a default value will be used based on the objective type. Please see the
            `default_ref_values` dict for these default values.

        Raises
        ------
            ValueError: If an invalid problem type is provided.

        """
        # Dictionary for default reference values
        default_ref_values = {
            'mass': -5e4,
            'hybrid_objective': -5e4,
            'fuel_burned': 1e4,
            'fuel': 1e4
        }

        # Check if an objective type is specified
        if objective_type is not None:
            ref = ref if ref is not None else default_ref_values.get(objective_type, 1)

            final_phase_name = self.regular_phases[-1]

            if objective_type == 'mass':
                if self.analysis_scheme is AnalysisScheme.COLLOCATION:
                    self.model.add_objective(
                        f"traj.{final_phase_name}.timeseries.{Dynamic.Vehicle.MASS}",
                        index=-1,
                        ref=ref
                    )
                else:
                    last_phase = self.traj._phases.items()[final_phase_name]
                    last_phase.add_objective(
                        Dynamic.Vehicle.MASS, loc='final', ref=ref)

            elif objective_type == 'time':
                self.model.add_objective(
                    f"traj.{final_phase_name}.timeseries.time", index=-1, ref=ref)

            elif objective_type == "hybrid_objective":
                self._add_hybrid_objective(self.phase_info)
                self.model.add_objective("obj_comp.obj")

            elif objective_type == "fuel_burned":
                self.model.add_objective(Mission.Summary.FUEL_BURNED, ref=ref)

            elif objective_type == "fuel":
                self.model.add_objective(Mission.Objectives.FUEL, ref=ref)

            else:
                raise ValueError(
                    f"{objective_type} is not a valid objective.\nobjective_type must"
                    " be one of mass, time, hybrid_objective, fuel_burned, or fuel")

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

    def _add_bus_variables_and_connect(self):
        all_subsystems = self._get_all_subsystems()

        base_phases = list(self.phase_info.keys())

        for external_subsystem in all_subsystems:
            bus_variables = external_subsystem.get_bus_variables()
            if bus_variables is not None:
                for bus_variable, variable_data in bus_variables.items():
                    mission_variable_name = variable_data['mission_name']

                    # check if mission_variable_name is a list
                    if not isinstance(mission_variable_name, list):
                        mission_variable_name = [mission_variable_name]

                    # loop over the mission_variable_name list and add each variable to
                    # the trajectory
                    for mission_var_name in mission_variable_name:
                        if mission_var_name not in self.meta_data:
                            # base_units = self.model.get_io_metadata(includes=f'pre_mission.{external_subsystem.name}.{bus_variable}')[f'pre_mission.{external_subsystem.name}.{bus_variable}']['units']
                            base_units = variable_data['units']

                            shape = variable_data.get('shape', _unspecified)

                            targets = mission_var_name
                            if '.' in mission_var_name:
                                # Support for non-hierarchy variables as parameters.
                                mission_var_name = mission_var_name.split('.')[-1]

                            if 'phases' in variable_data:
                                # Support for connecting bus variables into a subset of
                                # phases.
                                for phase_name in variable_data['phases']:
                                    phase = getattr(self.traj.phases, phase_name)

                                    phase.add_parameter(
                                        mission_var_name, opt=False, static_target=True,
                                        units=base_units, shape=shape, targets=targets)

                                    self.model.connect(
                                        f'pre_mission.{bus_variable}',
                                        f'traj.{phase_name}.parameters:{mission_var_name}')

                            else:
                                self.traj.add_parameter(
                                    mission_var_name, opt=False, static_target=True,
                                    units=base_units, shape=shape,
                                    targets={phase_name: [mission_var_name]
                                             for phase_name in base_phases})

                                self.model.connect(
                                    f'pre_mission.{bus_variable}',
                                    f'traj.parameters:' + mission_var_name)

                        if 'post_mission_name' in variable_data:
                            self.model.connect(
                                f'pre_mission.{external_subsystem.name}.{bus_variable}',
                                f'post_mission.{external_subsystem.name}.'
                                f'{variable_data["post_mission_name"]}',
                            )

    def setup(self, **kwargs):
        """
        Lightly wrapped setup() method for the problem.
        """
        # Use OpenMDAO's model options to pass all options through the system hierarchy.
        setup_model_options(self, self.aviary_inputs, self.meta_data)

        # suppress warnings:
        # "input variable '...' promoted using '*' was already promoted using 'aircraft:*'
        with warnings.catch_warnings():

            self.model.options['aviary_options'] = self.aviary_inputs
            self.model.options['aviary_metadata'] = self.meta_data
            self.model.options['phase_info'] = self.phase_info

            warnings.simplefilter("ignore", om.OpenMDAOWarning)
            warnings.simplefilter("ignore", om.PromotionWarning)

            super().setup(**kwargs)

    def set_initial_guesses(self, parent_prob=None, parent_prefix=""):
        """
        Call `set_val` on the trajectory for states and controls to seed
        the problem with reasonable initial guesses. This is especially
        important for collocation methods.
        This method first identifies all phases in the trajectory then
        loops over each phase. Specific initial guesses
        are added depending on the phase and mission method. Cruise is treated
        as a special phase for GASP-based missions because it is an AnalyticPhase
        in Dymos. For this phase, we handle the initial guesses first separately
        and continue to the next phase after that. For other phases, we set the initial
        guesses for states and controls according to the information available
        in the 'initial_guesses' attribute of the phase.
        """
        target_prob = self
        if parent_prob is not None and parent_prefix != "":
            target_prob = parent_prob

        # Grab the trajectory object from the model
        if self.analysis_scheme is AnalysisScheme.SHOOTING:
            if self.problem_type is ProblemType.SIZING:
                target_prob.set_val(parent_prefix + Mission.Summary.GROSS_MASS,
                                   self.get_val(Mission.Design.GROSS_MASS))

            target_prob.set_val(parent_prefix +
                               "traj.SGMClimb_" + Dynamic.Mission.ALTITUDE + "_trigger",
                               val=self.cruise_alt,
                               units="ft",
                               )

            return

        traj = self.model.traj

        # Determine which phases to loop over, fetching them from the trajectory
        phase_items = traj._phases.items()

        # Loop over each phase and set initial guesses for the state and control
        # variables
        for idx, (phase_name, phase) in enumerate(phase_items):

            if self.mission_method is SOLVED_2DOF:
                self.phase_objects[idx].apply_initial_guesses(self, 'traj', phase)
                if self.phase_info[phase_name]['user_options']['ground_roll'] and self.phase_info[phase_name]['user_options']['fix_initial']:
                    continue

            # If not, fetch the initial guesses specific to the phase
            # check if guesses exist for this phase
            if "initial_guesses" in self.phase_info[phase_name]:
                guesses = self.phase_info[phase_name]['initial_guesses']
            else:
                guesses = {}

            # Add subsystem guesses
            self._add_subsystem_guesses(phase_name, phase, target_prob, parent_prefix)

            # Set initial guesses for states and controls for each phase
            self.builder.add_guesses(self, phase_name, phase, guesses, target_prob,
                                     parent_prefix)

    def _process_guess_var(self, val, key, phase):
        """
        Process the guess variable, which can either be a float or an array of floats.
        This method is responsible for interpolating initial guesses when the user
        provides a list or array of values rather than a single float. It interpolates
        the guess values across the phase's domain for a given variable, be it a control
        or a state variable. The interpolation is performed between -1 and 1 (representing
        the normalized phase time domain), using the numpy linspace function.
        The result of this method is a single value or an array of interpolated values
        that can be used to seed the optimization problem with initial guesses.
        Parameters
        ----------
        val : float or list/array of floats
            The initial guess value(s) for a particular variable.
        key : str
            The key identifying the variable for which the initial guess is provided.
        phase : Phase
            The phase for which the variable is being set.
        Returns
        -------
        val : float or array of floats
            The processed guess value(s) to be used in the optimization problem.
        """
        # Check if val is not a single float
        if not isinstance(val, float):
            # If val is an array of values
            if len(val) > 1:
                # Get the shape of the val array
                shape = np.shape(val)

                # Generate an array of evenly spaced values between -1 and 1,
                # reshaping to match the shape of the val array
                xs = np.linspace(-1, 1, num=np.prod(shape)).reshape(shape)

                # Check if the key indicates a control or state variable
                if "controls:" in key or "states:" in key:
                    # If so, strip the first part of the key to match the variable name
                    # in phase
                    stripped_key = ":".join(key.split(":")[1:])

                    # Interpolate the initial guess values across the phase's domain
                    val = phase.interp(stripped_key, xs=xs, ys=val)
                else:
                    # If not a control or state variable, interpolate the initial guess
                    # values directly
                    val = phase.interp(key, xs=xs, ys=val)

        # Return the processed guess value(s)
        return val

    def _add_subsystem_guesses(self, phase_name, phase, target_prob, parent_prefix):
        """
        Adds the initial guesses for each subsystem of a given phase to the problem.
        This method first fetches all subsystems associated with the given phase.
        It then loops over each subsystem and fetches its initial guesses. For each
        guess, it identifies whether the guess corresponds to a state or a control
        variable and then processes the guess variable. After this, the initial
        guess is set in the problem using the `set_val` method.
        Parameters
        ----------
        phase_name : str
            The name of the phase for which the subsystem guesses are being added.
        phase : Phase
            The phase object for which the subsystem guesses are being added.
        """

        # Get all subsystems associated with the phase
        all_subsystems = self._get_all_subsystems(
            self.phase_info[phase_name]['external_subsystems'])

        # Loop over each subsystem
        for subsystem in all_subsystems:
            # Fetch the initial guesses for the subsystem
            initial_guesses = subsystem.get_initial_guesses()

            # Loop over each guess
            for key, val in initial_guesses.items():
                # Identify the type of the guess (state or control)
                type = val.pop('type')
                if 'state' in type:
                    path_string = 'states'
                elif 'control' in type:
                    path_string = 'controls'

                # Process the guess variable (handles array interpolation)
                val['val'] = self._process_guess_var(val['val'], key, phase)

                # Set the initial guess in the problem
                target_prob.set_val(
                    parent_prefix + f'traj.{phase_name}.{path_string}:{key}', **val)

    def run_aviary_problem(self, record_filename="problem_history.db",
                           optimization_history_filename=None, restart_filename=None,
                           suppress_solver_print=True, run_driver=True, simulate=False,
                           make_plots=True):
        """
        This function actually runs the Aviary problem, which could be a simulation, optimization, or a driver execution, depending on the arguments provided.

        Parameters
        ----------
        record_filename : str, optional
            The name of the database file where the solutions are to be recorded. The default is "problem_history.db".
        optimization_history_filename : str, None
            The name of the database file where the driver iterations are to be recorded. The default is None.
        restart_filename : str, optional
            The name of the file that contains previously computed solutions which are to be used as starting points for this run. If it is None (default), no restart file will be used.
        suppress_solver_print : bool, optional
            If True (default), all solvers' print statements will be suppressed. Useful for deeply nested models with multiple solvers so the print statements don't overwhelm the output.
        run_driver : bool, optional
            If True (default), the driver (aka optimizer) will be executed. If False, the problem will be run through one pass -- equivalent to OpenMDAO's `run_model` behavior.
        simulate : bool, optional
            If True, an explicit Dymos simulation will be performed. The default is False.
        make_plots : bool, optional
            If True (default), Dymos html plots will be generated as part of the output.
        """

        if self.aviary_inputs.get_val(Settings.VERBOSITY).value >= 2:
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
                self, run_driver=run_driver, simulate=simulate, make_plots=make_plots,
                solution_record_file=record_filename, restart=restart_filename)
        else:
            # prevent UserWarning that is displayed when an event is triggered
            warnings.filterwarnings('ignore', category=UserWarning)
            failed = self.run_model()
            warnings.filterwarnings('default', category=UserWarning)

        if self.aviary_inputs.get_val(Settings.VERBOSITY).value >= 2:
            with open('output_list.txt', 'w') as outfile:
                self.model.list_outputs(out_stream=outfile)

        self.problem_ran_successfully = not failed

    def alternate_mission(self, run_mission=True,
                          json_filename='sizing_problem.json',
                          payload_mass=None, mission_range=None,
                          phase_info=None, verbosity=Verbosity.BRIEF):
        """
        This function runs an alternate mission based on a sizing mission output.

        Parameters
        ----------
        run_mission : bool
            Flag to determine whether to run the mission before returning the problem object.
        json_filename : str
            Name of the file that the sizing mission has been saved to.
        mission_range : float, optional
            Target range for the fallout mission.
        payload_mass : float, optional
            Mass of the payload for the mission.
        phase_info : dict, optional
            Dictionary containing the phases and their required parameters.
        verbosity : Verbosity or list, optional
            If Verbosity.DEBUG, debug print options ['desvars','ln_cons','nl_cons','objs'] will be set.
            If a list is provided, it will be used as the debug print options.
        """
        if phase_info is None:
            phase_info = self.phase_info
        if mission_range is None:
            design_range = self.get_val(Mission.Design.RANGE)
        if payload_mass is None:
            if self.mission_method is HEIGHT_ENERGY:
                payload_mass = self.get_val(Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS)
            elif self.mission_method is TWO_DEGREES_OF_FREEDOM:
                payload_mass = self.get_val(Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS)

        mission_mass = self.get_val(Mission.Design.GROSS_MASS)
        optimizer = self.driver.options["optimizer"]

        prob_alternate = _load_off_design(
            json_filename,
            ProblemType.ALTERNATE,
            phase_info,
            payload_mass,
            design_range,
            mission_mass)

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
            prob_alternate.run_aviary_problem(
                record_filename='alternate_problem_history.db')
        return prob_alternate

    def fallout_mission(self, run_mission=True,
                        json_filename='sizing_problem.json',
                        mission_mass=None, payload_mass=None,
                        phase_info=None, verbosity=Verbosity.BRIEF):
        """
        This function runs a fallout mission based on a sizing mission output.

        Parameters
        ----------
        run_mission : bool
            Flag to determine whether to run the mission before returning the problem object.
        json_filename : str
            Name of the file that the sizing mission has been saved to.
        mission_mass : float, optional
            Takeoff mass for the fallout mission.
        payload_mass : float, optional
            Mass of the payload for the mission.
        phase_info : dict, optional
            Dictionary containing the phases and their required parameters.
        verbosity : Verbosity or list, optional
            If Verbosity.DEBUG, debug print options ['desvars','ln_cons','nl_cons','objs'] will be set.
            If a list is provided, it will be used as the debug print options.
        """
        if phase_info is None:
            phase_info = self.phase_info
        if mission_mass is None:
            mission_mass = self.get_val(Mission.Design.GROSS_MASS)
        if payload_mass is None:
            if self.mission_method is HEIGHT_ENERGY:
                payload_mass = self.get_val(Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS)
            elif self.mission_method is TWO_DEGREES_OF_FREEDOM:
                payload_mass = self.get_val(Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS)

        design_range = self.get_val(Mission.Design.RANGE)
        optimizer = self.driver.options["optimizer"]

        prob_fallout = _load_off_design(json_filename, ProblemType.FALLOUT, phase_info,
                                        payload_mass, design_range, mission_mass)

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
        aviary_problem: OpenMDAO Aviary Problem
            Aviary problem object optimized for the aircraft design/sizing mission.
            Assumed to contain aviary_inputs and Mission.Summary.GROSS_MASS
        json_filename:   string
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
                        Mission.Summary.GROSS_MASS, units=units)
                    Mission_Summary_GROSS_MASS_val_list = Mission_Summary_GROSS_MASS_val.tolist()
                    value = Mission_Summary_GROSS_MASS_val_list[0]

                else:
                    # there are different data types we need to handle for conversion to json format
                    # int, bool, float doesn't need anything special

                    # Convert numpy arrays to lists
                    if type_value == np.ndarray:
                        value = value.tolist()

                    # Lists are fine except if they contain enums
                    if type_value == list:
                        if isinstance(value[0], enum.Enum):
                            for i in range(len(value)):
                                value[i] = str([value[i]])

                    # Enums need converting to a string
                    if isinstance(value, enum.Enum):
                        value = str([value])

                # Append the data to the list
                aviary_input_list.append([name, value, units, str(type_value)])

            # Write the list to a json file
            json.dump(aviary_input_list, jsonfile, sort_keys=True,
                      indent=4, ensure_ascii=False)

            jsonfile.close()

    def _add_hybrid_objective(self, phase_info):
        phases = list(phase_info.keys())
        takeoff_mass = self.aviary_inputs.get_val(
            Mission.Design.GROSS_MASS, units='lbm')

        obj_comp = om.ExecComp(f"obj = -final_mass / {takeoff_mass} + final_time / 5.",
                               final_mass={"units": "lbm"},
                               final_time={"units": "h"})
        self.model.add_subsystem("obj_comp", obj_comp)

        final_phase_name = phases[-1]
        self.model.connect(f"traj.{final_phase_name}.timeseries.mass",
                           "obj_comp.final_mass", src_indices=[-1])
        self.model.connect(f"traj.{final_phase_name}.timeseries.time",
                           "obj_comp.final_time", src_indices=[-1])

    def _add_vrotate_comp(self):
        self.model.add_subsystem("vrot_comp", VRotateComp())
        self.model.connect('traj.groundroll.states:mass',
                           'vrot_comp.mass', src_indices=om.slicer[0, ...])

        vrot_eq_comp = self.model.add_subsystem("vrot_eq_comp", om.EQConstraintComp())
        vrot_eq_comp.add_eq_output(
            "v_rotate_error", eq_units="kn", lhs_name="v_rot_computed",
            rhs_name="groundroll_v_final", add_constraint=True)

        self.model.connect('vrot_comp.Vrot', 'vrot_eq_comp.v_rot_computed')
        self.model.connect(
            'traj.groundroll.timeseries.velocity', 'vrot_eq_comp.groundroll_v_final',
            src_indices=om.slicer[-1, ...])

    def _save_to_csv_file(self, filename):
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['name', 'value', 'units']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            for name, value_units in sorted(self.aviary_inputs):
                value, units = value_units
                writer.writerow({'name': name, 'value': value, 'units': units})

    def _get_all_subsystems(self, external_subsystems=None):
        all_subsystems = []
        if external_subsystems is None:
            all_subsystems.extend(self.pre_mission_info['external_subsystems'])
        else:
            all_subsystems.extend(external_subsystems)

        all_subsystems.append(self.core_subsystems['aerodynamics'])
        all_subsystems.append(self.core_subsystems['propulsion'])

        return all_subsystems

    def _add_comps_for_objs_and_cons(self):
        "add objectives and some constraints"
        self.model.add_subsystem(
            "fuel_obj",
            om.ExecComp(
                "reg_objective = overall_fuel/10000 + ascent_duration/30.",
                reg_objective={"val": 0.0, "units": "unitless"},
                ascent_duration={"units": "s", "shape": 1},
                overall_fuel={"units": "lbm"},
            ),
            promotes_inputs=[
                ("ascent_duration", Mission.Takeoff.ASCENT_DURATION),
                ("overall_fuel", Mission.Summary.TOTAL_FUEL_MASS),
            ],
            promotes_outputs=[("reg_objective", Mission.Objectives.FUEL)],
        )

        self.model.add_subsystem(
            "range_obj",
            om.ExecComp(
                "reg_objective = -actual_range/1000 + ascent_duration/30.",
                reg_objective={"val": 0.0, "units": "unitless"},
                ascent_duration={"units": "s", "shape": 1},
                actual_range={
                    "val": self.target_range, "units": "NM"},
            ),
            promotes_inputs=[
                ("actual_range", Mission.Summary.RANGE),
                ("ascent_duration", Mission.Takeoff.ASCENT_DURATION),
            ],
            promotes_outputs=[("reg_objective", Mission.Objectives.RANGE)],
        )

        if self.analysis_scheme is AnalysisScheme.COLLOCATION:
            if self.mission_method is TWO_DEGREES_OF_FREEDOM:
                ascent_phase = getattr(self.traj.phases, 'ascent')
                ascent_tx = ascent_phase.options["transcription"]
                ascent_num_nodes = ascent_tx.grid_data.num_nodes
                self.model.add_subsystem(
                    "h_fit",
                    PolynomialFit(N_cp=ascent_num_nodes),
                    promotes_inputs=["t_init_gear", "t_init_flaps"],
                )

        self.model.add_subsystem(
            "range_constraint",
            om.ExecComp(
                "range_resid = target_range - actual_range",
                target_range={"val": self.target_range, "units": "NM"},
                actual_range={"val": self.target_range, "units": "NM"},
                range_resid={"val": 30, "units": "NM"},
            ),
            promotes_inputs=[
                ("actual_range", Mission.Summary.RANGE),
                "target_range",
            ],
            promotes_outputs=[
                ("range_resid", Mission.Constraints.RANGE_RESIDUAL)],
        )

    def _add_fuel_reserve_component(self, post_mission=True,
                                    reserves_name=Mission.Design.RESERVE_FUEL):
        if post_mission:
            reserve_calc_location = self.post_mission
        else:
            reserve_calc_location = self.model

        RESERVE_FUEL_FRACTION = self.aviary_inputs.get_val(
            Aircraft.Design.RESERVE_FUEL_FRACTION, units='unitless')
        if RESERVE_FUEL_FRACTION != 0:
            reserve_fuel_frac = om.ExecComp(
                'reserve_fuel_frac_mass = reserve_fuel_fraction * (takeoff_mass - final_mass)',
                reserve_fuel_frac_mass={
                    "units": "lbm"},
                reserve_fuel_fraction={
                    "units": "unitless",
                    "val": RESERVE_FUEL_FRACTION},
                final_mass={
                    "units": "lbm"},
                takeoff_mass={
                    "units": "lbm"})

            reserve_calc_location.add_subsystem(
                "reserve_fuel_frac", reserve_fuel_frac,
                promotes_inputs=[("takeoff_mass", Mission.Summary.GROSS_MASS),
                                 ("final_mass", Mission.Landing.TOUCHDOWN_MASS),
                                 ("reserve_fuel_fraction", Aircraft.Design.
                                  RESERVE_FUEL_FRACTION)],
                promotes_outputs=["reserve_fuel_frac_mass"])

        RESERVE_FUEL_ADDITIONAL = self.aviary_inputs.get_val(
            Aircraft.Design.RESERVE_FUEL_ADDITIONAL, units='lbm')
        reserve_fuel = om.ExecComp(
            'reserve_fuel = reserve_fuel_frac_mass + reserve_fuel_additional + reserve_fuel_burned',
            reserve_fuel={"units": "lbm", 'shape': 1},
            reserve_fuel_frac_mass={"units": "lbm", "val": 0},
            reserve_fuel_additional={"units": "lbm", "val": RESERVE_FUEL_ADDITIONAL},
            reserve_fuel_burned={"units": "lbm", "val": 0})

        reserve_calc_location.add_subsystem(
            "reserve_fuel",
            reserve_fuel,
            promotes_inputs=[
                "reserve_fuel_frac_mass",
                ("reserve_fuel_additional",
                 Aircraft.Design.RESERVE_FUEL_ADDITIONAL),
                ("reserve_fuel_burned",
                 Mission.Summary.RESERVE_FUEL_BURNED)],
            promotes_outputs=[
                ("reserve_fuel",
                 reserves_name)])


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
    ----------
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

        # Initialize some flags to idetify arrays and enums
        is_array = False
        is_enum = False

        if var_type == "<class 'numpy.ndarray'>":
            is_array = True

        elif var_type == "<class 'list'>":
            # check if the list contains enums
            for i in range(len(var_values)):
                if isinstance(var_values[i], str):
                    if var_values[i].find("<") != -1:
                        # Found a list of enums: set the flag
                        is_enum = True

                        # Manipulate the string to find the value
                        tmp_var_values = var_values[i].split(':')[-1]
                        var_values[i] = tmp_var_values.replace(">", "").replace(
                            "]", "").replace("'", "").replace(" ", "")

            if is_enum:
                var_values = convert_strings_to_data(var_values)

            else:
                var_values = [var_values]

        elif var_type.find("<enum") != -1:
            # Identify enums and manipulate the string to find the value
            tmp_var_values = var_values.split(':')[-1]
            var_values = tmp_var_values.replace(">", "").replace(
                "]", "").replace("'", "").replace(" ", "")
            var_values = convert_strings_to_data([var_values])

        else:
            # values are expected to be parsed as a list to set_value function
            var_values = [var_values]

        # Check if the variable is in meta data
        if var_name in BaseMetaData.keys():
            try:
                aviary_problem.aviary_inputs = set_value(
                    var_name, var_values, aviary_problem.aviary_inputs, units=var_units,
                    is_array=is_array, meta_data=BaseMetaData)
            except BaseException:
                # Print helpful error
                print(
                    "FAILURE: list_num = ",
                    counter,
                    "Input String = ",
                    inputs,
                    "Attempted to set_value(",
                    var_name,
                    ",",
                    var_values,
                    ",",
                    var_units,
                    ")")
        else:
            # Not in the MetaData
            print(
                "Name not found in MetaData: list_num =",
                counter,
                "Input String =",
                inputs,
                "Attempted set_value(",
                var_name,
                ",",
                var_values,
                ",",
                var_units,
                ")")

        counter = counter + 1  # increment index tracker
    return aviary_problem


def _load_off_design(json_filename, ProblemType, phase_info,
                     payload, mission_range, mission_gross_mass):
    """
    This function loads a sized aircraft, and sets up an aviary problem
    for a specified off design mission.

    Parameters
    ----------
    json_filename:      string
        User specified name and relative path of json file containing the sized aircraft data
    ProblemType:        enum
        Alternate or Fallout. Alternate requires mission_range input and
         Fallout requires mission_fuel input
    phase_info:     phase_info dictionary for off design mission
    payload:            float
        Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS
    mission_range       float
        Mission.Summary.RANGE 'NM'
    mission_gross_mass  float
        Mission.Summary.GROSS_MASS 'lbm'

    Returns
    ----------
    Aviary Problem object with completed load_inputs() for specified off design mission
    """

    # Initialize a new aviary problem and aviary_input data structure
    prob = AviaryProblem()
    prob.aviary_inputs = AviaryValues()

    prob = _read_sizing_json(prob, json_filename)

    # Update problem type
    prob.problem_type = ProblemType
    prob.aviary_inputs.set_val('settings:problem_type', ProblemType, units='unitless')

    # Set Payload
    prob.aviary_inputs.set_val(
        Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS, payload, units='lbm')

    if ProblemType == ProblemType.ALTERNATE:
        # Set mission range, aviary will calculate required fuel
        prob.aviary_inputs.set_val(Mission.Design.RANGE, mission_range, units='NM')

    elif ProblemType == ProblemType.FALLOUT:
        # Set mission fuel and calculate gross weight, aviary will calculate range
        prob.aviary_inputs.set_val(Mission.Summary.GROSS_MASS,
                                   mission_gross_mass, units='lbm')

    # Load inputs
    prob.load_inputs(prob.aviary_inputs, phase_info)
    return prob
