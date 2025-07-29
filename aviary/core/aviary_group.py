import inspect
from pathlib import Path
from importlib.machinery import SourceFileLoader

import dymos as dm
from dymos.utils.misc import _unspecified
import openmdao.api as om
from openmdao.utils.mpi import MPI

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import EquationsOfMotion
from aviary.variable_info.variables import Settings
from aviary.variable_info.enums import Verbosity
from aviary.core.pre_mission_group import PreMissionGroup
from aviary.core.post_mission_group import PostMissionGroup
from aviary.utils.preprocessors import preprocess_options
from aviary.variable_info.enums import (
    EquationsOfMotion,
    LegacyCode,
    ProblemType,
    Verbosity,
)
from aviary.mission.height_energy_problem_configurator import HeightEnergyProblemConfigurator
from aviary.mission.solved_two_dof_problem_configurator import SolvedTwoDOFProblemConfigurator
from aviary.mission.two_dof_problem_configurator import TwoDOFProblemConfigurator
from aviary.subsystems.aerodynamics.aerodynamics_builder import CoreAerodynamicsBuilder
from aviary.subsystems.geometry.geometry_builder import CoreGeometryBuilder
from aviary.subsystems.mass.mass_builder import CoreMassBuilder
from aviary.subsystems.premission import CorePreMission
from aviary.subsystems.propulsion.propulsion_builder import CorePropulsionBuilder
from aviary.interface.utils import set_warning_format
from aviary.mission.utils import get_phase_mission_bus_lengths, process_guess_var
from aviary.variable_info.variables import Aircraft, Dynamic, Mission, Settings
from aviary.variable_info.variable_meta_data import _MetaData as BaseMetaData

from aviary.utils.functions import get_path
from aviary.utils.preprocessors import preprocess_options
from aviary.utils.process_input_decks import create_vehicle, update_GASP_options
from aviary.utils.utils import wrapped_convert_units
from aviary.variable_info.functions import setup_trajectory_params

TWO_DEGREES_OF_FREEDOM = EquationsOfMotion.TWO_DEGREES_OF_FREEDOM
HEIGHT_ENERGY = EquationsOfMotion.HEIGHT_ENERGY
SOLVED_2DOF = EquationsOfMotion.SOLVED_2DOF
CUSTOM = EquationsOfMotion.CUSTOM

FLOPS = LegacyCode.FLOPS
GASP = LegacyCode.GASP


class AviaryGroup(om.Group):
    """
    A standard OpenMDAO group where all elements of a given aviary aircraft design and mission are
    defined.

    This includes pre_mission, mission, and post_mission analysis. This group also contains methods
    for loading data from .csv and phase_info files, setting initial values on the group, and
    connecting all the phases inside the mission analysis to each other.

    Instantiating multiple AviaryGroups allows for analysis and optimization of multiple aircraft or
    one aircraft in multiple missions simultaneously.
    """

    def __init__(self, verbosity=None, **kwargs):
        super().__init__(**kwargs)

        self.pre_mission = PreMissionGroup()
        self.post_mission = PostMissionGroup()
        self.verbosity = verbosity
        self.regular_phases = []
        self.reserve_phases = []

    def initialize(self):
        """Declare options."""
        self.options.declare(
            'aviary_options',
            types=AviaryValues,
            desc='collection of Aircraft/Mission specific options',
        )
        self.options.declare(
            'aviary_metadata', types=dict, desc='metadata dictionary of the full aviary problem.'
        )
        self.options.declare('phase_info', types=dict, desc='phase-specific settings.')
        self.builder = []

    def configure(self):
        """Configure the Aviary group."""
        aviary_options = self.options['aviary_options']
        aviary_metadata = self.options['aviary_metadata']

        # Find promoted name of every input in the model.
        all_prom_inputs = []

        # We can call list_inputs on the subsystems.
        for system in self.system_iter(recurse=False):
            var_abs = system.list_inputs(out_stream=None, val=False)
            var_prom = [v['prom_name'] for k, v in var_abs]
            all_prom_inputs.extend(var_prom)

            # Calls to promotes aren't handled until this group resolves.
            # Here, we address anything promoted with an alias in AviaryProblem.
            input_meta = system._var_promotes['input']
            var_prom = [v[0][1] for v in input_meta if isinstance(v[0], tuple)]
            all_prom_inputs.extend(var_prom)
            var_prom = [v[0] for v in input_meta if not isinstance(v[0], tuple)]
            all_prom_inputs.extend(var_prom)

        if MPI and self.comm.size > 1:
            # Under MPI, promotion info only lives on rank 0, so broadcast.
            all_prom_inputs = self.comm.bcast(all_prom_inputs, root=0)

        for key in aviary_metadata:
            if ':' not in key or key.startswith('dynamic:'):
                continue

            if aviary_metadata[key]['option']:
                continue

            # Skip anything that is not presently an input.
            if key not in all_prom_inputs:
                continue

            if key in aviary_options:
                val, units = aviary_options.get_item(key)
            else:
                val = aviary_metadata[key]['default_value']
                units = aviary_metadata[key]['units']

                if val is None:
                    # optional, but no default value
                    continue

            self.set_input_defaults(key, val=val, units=units)

        # try to get all the possible EOMs from the Enums rather than specifically calling the names here
        # This will require some modifications to the enums
        mission_method = aviary_options.get_val(Settings.EQUATIONS_OF_MOTION)

        # Temporarily add extra stuff here, probably patched soon
        if mission_method is HEIGHT_ENERGY:
            phase_info = self.options['phase_info']

            # Set a more appropriate solver for dymos when the phases are linked.
            if MPI and isinstance(self.traj.phases.linear_solver, om.PETScKrylov):
                # When any phase is connected with input_initial = True, dymos puts
                # a jacobi solver in the phases group. This is necessary in case
                # the phases are cyclic. However, this causes some problems
                # with the newton solvers in Aviary, exacerbating issues with
                # solver tolerances at multiple levels. Since Aviary's phases
                # are basically in series, the jacobi solver is a much better
                # choice and should be able to handle it in a couple of
                # iterations.
                self.traj.phases.linear_solver = om.LinearBlockJac(maxiter=5)

            # Due to recent changes in dymos, there is now a solver in any phase
            # that has connected initial states. It is not clear that this solver
            # is necessary except in certain corner cases that do not apply to the
            # Aviary trajectory. In our case, this solver merely addresses a lag
            # in the state input component. Since this solver can cause some
            # numerical problems, and can slow things down, we need to move it down
            # into the state interp component.
            # TODO: Future updates to dymos may make this unnecessary.
            for phase in self.traj.phases.system_iter(recurse=False):
                # Don't move the solvers if we are using solve segments.
                if phase_info[phase.name]['user_options'].get('distance_solve_segments'):
                    continue

                phase.nonlinear_solver = om.NonlinearRunOnce()
                phase.linear_solver = om.LinearRunOnce()
                if isinstance(phase.indep_states, om.ImplicitComponent):
                    phase.indep_states.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
                    phase.indep_states.linear_solver = om.DirectSolver(rhs_checking=True)

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
        This method loads the aviary_values inputs and options that the
        user specifies. They could specify files to load and values to
        replace here as well.
        Phase info is also loaded if provided by the user. If phase_info is None,
        the appropriate default phase_info based on mission analysis method is used.

        This method is not strictly necessary; a user could also supply
        an AviaryValues object and/or phase_info dict of their own.
        """
        ## LOAD INPUT FILE ###
        # Create AviaryValues object from file (or process existing AviaryValues object
        # with default values from metadata) and generate initial guesses
        aviary_inputs, self.initialization_guesses = create_vehicle(
            aircraft_data, meta_data=meta_data, verbosity=verbosity
        )

        # Update default verbosity now that we have read the input data, if a global verbosity
        # override was not requested
        if self.verbosity is None:
            self.verbosity = aviary_inputs.get_val(Settings.VERBOSITY)
            # set default warning format for the rest of the problem
            set_warning_format(self.verbosity)

        # If user did not ask for verbosity override for this method either, use the problem's
        # default verbosity for the rest of the method
        if verbosity is None:
            verbosity = self.verbosity
        # Now that the input file has been read, we have the desired verbosity for this
        # run stored in aviary_inputs. Save this to self.
        self.aviary_inputs = aviary_inputs

        # pull which methods will be used for subsystems and mission
        self.mission_method = mission_method = aviary_inputs.get_val(Settings.EQUATIONS_OF_MOTION)
        self.mass_method = mass_method = aviary_inputs.get_val(Settings.MASS_METHOD)
        self.aero_method = aero_method = aviary_inputs.get_val(Settings.AERODYNAMICS_METHOD)

        # Create engine_builder
        self.engine_builders = engine_builders

        # Determine which problem configurator to use based on mission_method
        if mission_method is HEIGHT_ENERGY:
            self.configurator = HeightEnergyProblemConfigurator()
        elif mission_method is TWO_DEGREES_OF_FREEDOM:
            self.configurator = TwoDOFProblemConfigurator()
        elif mission_method is SOLVED_2DOF:
            self.configurator = SolvedTwoDOFProblemConfigurator()
        elif mission_method is CUSTOM:
            if problem_configurator:
                self.configurator = problem_configurator()
                # TODO: make draft / example custom builder
            else:
                raise ValueError(
                    'When using "settings:equations_of_motion,custom", a '
                    'problem_configurator must be specified in load_inputs().'
                )
        else:
            raise ValueError(
                'settings:equations_of_motion must be one of: height_energy, 2DOF, '
                'solved_2DOF, or custom'
            )

        # TODO this should be a preprocessor step if it is required here
        if mass_method is GASP or aero_method is GASP:
            aviary_inputs = update_GASP_options(aviary_inputs)

        ## LOAD PHASE_INFO ###
        # if phase info is a file, load it
        if isinstance(phase_info, str) or isinstance(phase_info, Path):
            phase_info_path = get_path(phase_info)
            phase_info_file = SourceFileLoader(
                'phase_info_file', str(phase_info_path)
            ).load_module()
            phase_info = getattr(phase_info_file, 'phase_info')

        if phase_info is None:
            phase_info = self.configurator.get_default_phase_info(self)
            if verbosity is not None and verbosity >= Verbosity.BRIEF:
                print(
                    f'Loaded default phase_info for {self.mission_method.value.lower()} equations '
                    'of motion.'
                )

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
            self.pre_mission_info = {}

        if 'post_mission' in phase_info:
            self.post_mission_info = phase_info['post_mission']
        else:
            self.post_mission_info = {}

        self.problem_type = aviary_inputs.get_val(Settings.PROBLEM_TYPE)

        self.configurator.initial_guesses(self)
        # This function sets all the following defaults if they were not already set
        # self.engine_builders, self.pre_mission_info, self_post_mission_info
        # self.require_range_residual, self.target_range
        # Other specific self.*** are defined in here as well that are specific to
        # each builder

        return self.aviary_inputs, self.verbosity

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

        aviary_inputs = self.aviary_inputs
        # Target_distance verification for all phases
        # Checks to make sure target_distance is positive,
        for phase_name, phase in self.phase_info.items():
            if 'user_options' in phase:
                target_distance = phase['user_options'].get('target_distance', (None, 'm'))
                if target_distance[0] is not None and target_distance[0] <= 0:
                    raise ValueError(
                        f'Invalid target_distance in [{phase_name}].[user_options]. '
                        f'Current (value: {target_distance[0]}), '
                        f'(units: {target_distance[1]}) <= 0'
                    )

        # Checks to make sure time_duration is positive,
        # Sets duration_bounds, initial_guesses, and fixed_duration
        for phase_name, phase in self.phase_info.items():
            if 'user_options' in phase:
                analytic = False
                if self.mission_method is EquationsOfMotion.TWO_DEGREES_OF_FREEDOM:
                    try:
                        # if the user provided an option, use it
                        analytic = phase['user_options']['analytic']
                    except KeyError:
                        # if it isn't specified, only the default 2DOF cruise for
                        # collocation is analytic
                        if 'cruise' in phase_name:
                            analytic = phase['user_options']['analytic'] = True
                        else:
                            analytic = phase['user_options']['analytic'] = False

                if 'time_duration' in phase['user_options']:
                    time_duration, units = phase['user_options']['time_duration']

                    if time_duration is None:
                        continue

                    if time_duration <= 0:
                        raise ValueError(
                            f'Invalid time_duration in phase_info[{phase_name}]'
                            f'[user_options]. Current (value: {time_duration[0]}), '
                            f'(units: {time_duration[1]}) <= 0")'
                        )

                    if 'initial_guesses' not in phase:
                        phase['initial_guesses'] = {}

                    guesses = phase['initial_guesses']
                    if 'time' in guesses:
                        time_guess, units_guess = guesses['time']

                        if time_guess[1] is not None:
                            msg = f'Duration initial guess of {time_guess[1]} {units_guess} '
                            msg += f'specified on fixed duration phase for phase {phase_name}. '
                            msg += f'Using fixed value of {time_duration} {units} instead.'
                            print(msg)

                        time_duration_conv = wrapped_convert_units(
                            (time_duration, units), units_guess
                        )
                        guesses['time'] = ((time_guess[0], time_duration_conv), units_guess)

                    else:
                        guesses['time'] = ((None, time_duration), units)

        for phase_name in self.phase_info:
            for external_subsystem in self.phase_info[phase_name]['external_subsystems']:
                aviary_inputs = external_subsystem.preprocess_inputs(aviary_inputs)

        # PREPROCESSORS #
        # BUG we can't provide updated metadata to preprocessors, because we need the
        #     processed options to build our subsystems to begin with
        preprocess_options(
            aviary_inputs,
            engine_models=self.engine_builders,
            verbosity=verbosity,
            # metadata=self.meta_data
        )

        ## Set Up Core Subsystems ##
        prop = CorePropulsionBuilder('core_propulsion', engine_models=self.engine_builders)
        mass = CoreMassBuilder('core_mass', code_origin=self.mass_method)

        # If all phases ask for tabular aero, we can skip pre-mission. Check phase_info
        tabular = False
        for phase in self.phase_info:
            if phase not in ('pre_mission', 'post_mission'):
                try:
                    if (
                        'tabular'
                        in self.phase_info[phase]['subsystem_options']['core_aerodynamics'][
                            'method'
                        ]
                    ):
                        tabular = True
                except KeyError:
                    tabular = False

        aero = CoreAerodynamicsBuilder(
            'core_aerodynamics', code_origin=self.aero_method, tabular=tabular
        )

        # which geometry methods should be used?
        geom_code_origin = None

        if (self.aero_method is FLOPS) and (self.mass_method is FLOPS):
            geom_code_origin = FLOPS
        elif (self.aero_method is GASP) and (self.mass_method is GASP):
            geom_code_origin = GASP
        else:
            geom_code_origin = (FLOPS, GASP)

        # which geometry method gets prioritized in case of conflicting outputs
        code_origin_to_prioritize = self.configurator.get_code_origin(self)

        geom = CoreGeometryBuilder(
            'core_geometry',
            code_origin=geom_code_origin,
            code_origin_to_prioritize=code_origin_to_prioritize,
        )

        subsystems = self.core_subsystems = {
            'propulsion': prop,
            'geometry': geom,
            'mass': mass,
            'aerodynamics': aero,
        }

        # TODO optionally accept which subsystems to load from phase_info
        default_mission_subsystems = [
            subsystems['aerodynamics'],
            subsystems['propulsion'],
        ]
        self.ode_args = {
            'aviary_options': aviary_inputs,
            'core_subsystems': default_mission_subsystems,
        }

        # self._update_metadata_from_subsystems()
        self._check_reserve_phase_separation()

    def _check_reserve_phase_separation(self):
        """
        This method checks for reserve=True & False
        Returns an error if a non-reserve phase is specified after a reserve phase.
        return two dictionaries of phases: regular_phases and reserve_phases
        For shooting trajectories, this will also check if a phase is part of the descent.
        """
        # Check to ensure no non-reserve phases are specified after reserve phases
        start_reserve = False
        raise_error = False
        for idx, phase_name in enumerate(self.phase_info):
            if 'user_options' in self.phase_info[phase_name]:
                if 'reserve' in self.phase_info[phase_name]['user_options']:
                    if self.phase_info[phase_name]['user_options']['reserve'] is False:
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
                'In phase_info, reserve=False cannot be specified after a phase where '
                'reserve=True. All reserve phases must happen after non-reserve phases. '
                # TODO: will need to pre-pend current group level to all error messages!!
                f'Regular Phases : {self.regular_phases} | '
                f'Reserve Phases : {self.reserve_phases} '
            )

    def add_pre_mission_systems(self, verbosity=None):
        """
        Add pre-mission systems to the Aviary group. These systems are executed before
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

        pre_mission = self.pre_mission
        self.add_subsystem(
            'pre_mission',
            pre_mission,
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=['aircraft:*', 'mission:*'],
        )

        if 'linear_solver' in self.pre_mission_info:
            pre_mission.linear_solver = self.pre_mission_info['linear_solver']

        if 'nonlinear_solver' in self.pre_mission_info:
            pre_mission.nonlinear_solver = self.pre_mission_info['nonlinear_solver']

        self._add_premission_external_subsystems()

        subsystems = self.core_subsystems

        # Propulsion isn't included in core pre-mission group to avoid override step in
        # configure() - instead add it now
        pre_mission.add_subsystem(
            'core_propulsion',
            subsystems['propulsion'].build_pre_mission(self.aviary_inputs),
        )

        default_subsystems = [
            subsystems['geometry'],
            subsystems['aerodynamics'],
            subsystems['mass'],
        ]

        pre_mission.add_subsystem(
            'core_subsystems',
            CorePreMission(
                aviary_options=self.aviary_inputs,
                subsystems=default_subsystems,
                process_overrides=False,
            ),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        if self.pre_mission_info['include_takeoff']:
            self.configurator.add_takeoff_systems(self)

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
            subsystem_premission = external_subsystem.build_pre_mission(self.aviary_inputs)

            if subsystem_premission is not None:
                self.pre_mission.add_subsystem(external_subsystem.name, subsystem_premission)

                mass_names.extend(external_subsystem.get_mass_names())

        if mass_names:
            formatted_names = []
            for name in mass_names:
                formatted_name = name.replace(':', '_')
                formatted_names.append(formatted_name)

            # Define the expression for computing the sum of masses
            expr = 'subsystem_mass = ' + ' + '.join(formatted_names)

            promotes_inputs_list = [
                (formatted_name, original_name)
                for formatted_name, original_name in zip(formatted_names, mass_names)
            ]

            # Create the ExecComp
            self.pre_mission.add_subsystem(
                'external_comp_sum',
                om.ExecComp(expr, units='kg'),
                promotes_inputs=promotes_inputs_list,
                promotes_outputs=[('subsystem_mass', Aircraft.Design.EXTERNAL_SUBSYSTEMS_MASS)],
            )

    def _get_phase(self, phase_name, phase_idx, comm):
        phase_options = self.phase_info[phase_name]

        # TODO optionally accept which subsystems to load from phase_info
        subsystems = self.core_subsystems
        default_mission_subsystems = [
            subsystems['aerodynamics'],
            subsystems['propulsion'],
        ]

        phase_builder = self.configurator.get_phase_builder(self, phase_name, phase_options)

        phase_object = phase_builder.from_phase_info(
            phase_name,
            phase_options,
            default_mission_subsystems,
            meta_data=self.meta_data,
        )

        phase = phase_object.build_phase(aviary_options=self.aviary_inputs)

        self.phase_objects.append(phase_object)

        # TODO: add logic to filter which phases get which controls.
        # right now all phases get all controls added from every subsystem.
        # for example, we might only want ELECTRIC_SHAFT_POWER applied during the
        # climb phase.
        all_subsystems = self.get_all_subsystems(phase_options['external_subsystems'])

        # loop through all_subsystems and call `get_controls` on each subsystem
        for subsystem in all_subsystems:
            # add the controls from the subsystems to each phase
            arg_spec = inspect.getfullargspec(subsystem.get_controls)
            if 'phase_name' in arg_spec.args:
                control_dicts = subsystem.get_controls(phase_name=phase_name)
            else:
                control_dicts = subsystem.get_controls(phase_name=phase_name)
            for control_name, control_dict in control_dicts.items():
                phase.add_control(control_name, **control_dict)

        # This fills in all defaults from the phase_builders user_options.
        full_options = phase_object.user_options.to_phase_info()
        self.phase_info[phase_name]['user_options'] = full_options

        # TODO: Should some of this stuff be moved into the phase builder?
        self.configurator.set_phase_options(self, phase_name, phase_idx, phase, full_options, comm)

        return phase

    def add_phases(
        self, phase_info_parameterization=None, parallel_phases=True, verbosity=None, comm=None
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
        traj: The Dymos Trajectory object containing the added mission phases.
        """
        # `self.verbosity` is "true" verbosity for entire run. `verbosity` is verbosity
        # override for just this method
        if verbosity is not None:
            # compatibility with being passed int for verbosity
            verbosity = Verbosity(verbosity)
        else:
            verbosity = self.verbosity  # defaults to BRIEF

        if phase_info_parameterization is not None:
            self.phase_info, self.post_mission_info = phase_info_parameterization(
                self.phase_info, self.post_mission_info, self.aviary_inputs
            )

        phase_info = self.phase_info

        phases = list(phase_info.keys())
        traj = self.add_subsystem('traj', dm.Trajectory(parallel_phases=parallel_phases))

        def add_subsystem_timeseries_outputs(phase, phase_name):
            phase_options = self.phase_info[phase_name]
            all_subsystems = self.get_all_subsystems(phase_options['external_subsystems'])
            for subsystem in all_subsystems:
                timeseries_to_add = subsystem.get_outputs()
                for timeseries in timeseries_to_add:
                    phase.add_timeseries_output(timeseries)
                mbvars = subsystem.get_post_mission_bus_variables(
                    self.aviary_inputs, self.phase_info
                )
                if mbvars:
                    mbvars_this_phase = mbvars.get(phase_name, None)
                    if mbvars_this_phase:
                        timeseries_to_add = mbvars_this_phase.keys()
                        for timeseries in timeseries_to_add:
                            phase.add_timeseries_output(
                                timeseries, timeseries='mission_bus_variables'
                            )

        self.phase_objects = []
        for phase_idx, phase_name in enumerate(phases):
            phase = traj.add_phase(phase_name, self._get_phase(phase_name, phase_idx, comm))
            add_subsystem_timeseries_outputs(phase, phase_name)

        # loop through phase_info and external subsystems
        external_parameters = {}
        for phase_name in self.phase_info:
            external_parameters[phase_name] = {}
            all_subsystems = self.get_all_subsystems(
                self.phase_info[phase_name]['external_subsystems']
            )

            subsystem_options = phase_info[phase_name].get('subsystem_options', {})

            for subsystem in all_subsystems:
                if subsystem.name in subsystem_options:
                    kwargs = subsystem_options[subsystem.name]
                else:
                    kwargs = {}
                parameter_dict = subsystem.get_parameters(
                    phase_info=self.phase_info[phase_name],
                    aviary_inputs=self.aviary_inputs,
                    **kwargs,
                )
                for parameter in parameter_dict:
                    external_parameters[phase_name][parameter] = parameter_dict[parameter]

        traj = setup_trajectory_params(
            self,
            traj,
            self.aviary_inputs,
            phases,
            meta_data=self.meta_data,
            external_parameters=external_parameters,
        )

        self.traj = traj

        return traj

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

        post_mission = self.post_mission
        self.add_subsystem(
            'post_mission',
            post_mission,
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        self.configurator.add_post_mission_systems(self)

        # Add all post-mission external subsystems.
        phase_mission_bus_lengths = get_phase_mission_bus_lengths(self.traj)
        for external_subsystem in self.post_mission_info['external_subsystems']:
            subsystem_postmission = external_subsystem.build_post_mission(
                aviary_inputs=self.aviary_inputs,
                phase_info=self.phase_info,
                phase_mission_bus_lengths=phase_mission_bus_lengths,
            )

            if subsystem_postmission is not None:
                post_mission.add_subsystem(external_subsystem.name, subsystem_postmission)

        # Check if regular_phases[] is accessible
        try:
            self.regular_phases[0]
        except BaseException:
            raise ValueError(
                'regular_phases[] dictionary is not accessible. For HEIGHT_ENERGY and '
                'SOLVED_2DOF missions, check_and_preprocess_inputs() must be called '
                'before add_post_mission_systems().'
            )

        # Fuel burn in regular phases
        ecomp = om.ExecComp(
            'fuel_burned = initial_mass - mass_final',
            initial_mass={'units': 'lbm'},
            mass_final={'units': 'lbm'},
            fuel_burned={'units': 'lbm'},
        )

        post_mission.add_subsystem(
            'fuel_burned',
            ecomp,
            promotes=[('fuel_burned', Mission.Summary.FUEL_BURNED)],
        )

        if self.pre_mission_info['include_takeoff']:
            post_mission.promotes(
                'fuel_burned',
                [('initial_mass', Mission.Summary.GROSS_MASS)],
            )
        else:
            # timeseries has to be used because Breguet cruise phases don't have
            # states
            self.connect(
                f'traj.{self.regular_phases[0]}.timeseries.mass',
                'fuel_burned.initial_mass',
                src_indices=[0],
            )

        self.connect(
            f'traj.{self.regular_phases[-1]}.timeseries.mass',
            'fuel_burned.mass_final',
            src_indices=[-1],
        )

        # Fuel burn in reserve phases
        if self.reserve_phases:
            ecomp = om.ExecComp(
                'reserve_fuel_burned = initial_mass - mass_final',
                initial_mass={'units': 'lbm'},
                mass_final={'units': 'lbm'},
                reserve_fuel_burned={'units': 'lbm'},
            )

            post_mission.add_subsystem(
                'reserve_fuel_burned',
                ecomp,
                promotes=[('reserve_fuel_burned', Mission.Summary.RESERVE_FUEL_BURNED)],
            )

            # timeseries has to be used because Breguet cruise phases don't have
            # states
            self.connect(
                f'traj.{self.reserve_phases[0]}.timeseries.mass',
                'reserve_fuel_burned.initial_mass',
                src_indices=[0],
            )
            self.connect(
                f'traj.{self.reserve_phases[-1]}.timeseries.mass',
                'reserve_fuel_burned.mass_final',
                src_indices=[-1],
            )

        self.add_fuel_reserve_component()

        # TODO: need to add some sort of check that this value is less than the fuel capacity
        # TODO: the overall_fuel variable is the burned fuel plus the reserve, but should
        # also include the unused fuel, and the hierarchy variable name should be
        # more clear
        ecomp = om.ExecComp(
            'overall_fuel = (1 + fuel_margin/100)*fuel_burned + reserve_fuel',
            overall_fuel={'units': 'lbm', 'shape': 1},
            fuel_margin={'units': 'unitless', 'val': 0},
            fuel_burned={'units': 'lbm'},  # from regular_phases only
            reserve_fuel={'units': 'lbm', 'shape': 1},
        )
        post_mission.add_subsystem(
            'fuel_calc',
            ecomp,
            promotes_inputs=[
                ('fuel_margin', Aircraft.Fuel.FUEL_MARGIN),
                ('fuel_burned', Mission.Summary.FUEL_BURNED),
                ('reserve_fuel', Mission.Design.RESERVE_FUEL),
            ],
            promotes_outputs=[('overall_fuel', Mission.Summary.TOTAL_FUEL_MASS)],
        )

        # If a target distance (or time) has been specified for this phase
        # distance (or time) is measured from the start of this phase to the end
        # of this phase
        for phase_name in self.phase_info:
            user_options = self.phase_info[phase_name]['user_options']

            target_distance = user_options.get('target_distance', (None, 'nmi'))
            target_distance = wrapped_convert_units(target_distance, 'nmi')
            if target_distance is not None:
                post_mission.add_subsystem(
                    f'{phase_name}_distance_constraint',
                    om.ExecComp(
                        'distance_resid = target_distance - (final_distance - initial_distance)',
                        distance_resid={'units': 'nmi'},
                        target_distance={'val': target_distance, 'units': 'nmi'},
                        final_distance={'units': 'nmi'},
                        initial_distance={'units': 'nmi'},
                    ),
                )
                self.connect(
                    f'traj.{phase_name}.timeseries.distance',
                    f'{phase_name}_distance_constraint.final_distance',
                    src_indices=[-1],
                )
                self.connect(
                    f'traj.{phase_name}.timeseries.distance',
                    f'{phase_name}_distance_constraint.initial_distance',
                    src_indices=[0],
                )
                self.add_constraint(
                    f'{phase_name}_distance_constraint.distance_resid',
                    equals=0.0,
                    ref=1e2,
                )

            # this is only used for analytic phases with a target duration
            time_duration = user_options.get('time_duration', (None, 'min'))
            time_duration = wrapped_convert_units(time_duration, 'min')
            analytic = user_options.get('analytic', False)

            if analytic and time_duration is not None:
                post_mission.add_subsystem(
                    f'{phase_name}_duration_constraint',
                    om.ExecComp(
                        'duration_resid = time_duration - (final_time - initial_time)',
                        duration_resid={'units': 'min'},
                        time_duration={'val': time_duration, 'units': 'min'},
                        final_time={'units': 'min'},
                        initial_time={'units': 'min'},
                    ),
                )
                self.connect(
                    f'traj.{phase_name}.timeseries.time',
                    f'{phase_name}_duration_constraint.final_time',
                    src_indices=[-1],
                )
                self.connect(
                    f'traj.{phase_name}.timeseries.time',
                    f'{phase_name}_duration_constraint.initial_time',
                    src_indices=[0],
                )
                self.add_constraint(
                    f'{phase_name}_duration_constraint.duration_resid',
                    equals=0.0,
                    ref=1e2,
                )

        ecomp = om.ExecComp(
            'mass_resid = operating_empty_mass + overall_fuel + payload_mass - initial_mass',
            operating_empty_mass={'units': 'lbm'},
            overall_fuel={'units': 'lbm'},
            payload_mass={'units': 'lbm'},
            initial_mass={'units': 'lbm'},
            mass_resid={'units': 'lbm'},
        )

        payload_mass_src = Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS

        post_mission.add_subsystem(
            'mass_constraint',
            ecomp,
            promotes_inputs=[
                ('operating_empty_mass', Aircraft.Design.OPERATING_MASS),
                ('overall_fuel', Mission.Summary.TOTAL_FUEL_MASS),
                ('payload_mass', payload_mass_src),
                ('initial_mass', Mission.Summary.GROSS_MASS),
            ],
            promotes_outputs=[('mass_resid', Mission.Constraints.MASS_RESIDUAL)],
        )

    def link_phases(self, verbosity=None, comm=None):
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
        # parallelize connected phases, but it requires a solver that we would like
        # to avoid.
        true_unless_mpi = True
        if comm.size > 1 and self.traj.options['parallel_phases']:
            true_unless_mpi = False

        # loop over unique variable names
        for var in unique_vars:
            phases_to_link = []
            for idx, phase_name in enumerate(self.phase_info):
                if var in lists_to_link[idx]:
                    phases_to_link.append(phase_name)

            if len(phases_to_link) > 1:  # TODO: hack
                self.traj.link_phases(phases=phases_to_link, vars=[var], connected=True)

        self.configurator.link_phases(self, phases, connect_directly=true_unless_mpi)

        self._connect_mission_bus_variables()

        self.configurator.check_trajectory(self)

    def _add_bus_variables_and_connect(self):
        all_subsystems = self.get_all_subsystems()

        base_phases = list(self.phase_info.keys())

        for external_subsystem in all_subsystems:
            bus_variables = external_subsystem.get_pre_mission_bus_variables(self.aviary_inputs)
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
                            # base_units = self.get_io_metadata(includes=f'pre_mission.{external_subsystem.name}.{bus_variable}')[f'pre_mission.{external_subsystem.name}.{bus_variable}']['units']
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
                                        mission_var_name,
                                        opt=False,
                                        static_target=True,
                                        units=base_units,
                                        shape=shape,
                                        targets=targets,
                                    )

                                    self.connect(
                                        f'pre_mission.{bus_variable}',
                                        f'traj.{phase_name}.parameters:{mission_var_name}',
                                    )

                            else:
                                self.traj.add_parameter(
                                    mission_var_name,
                                    opt=False,
                                    static_target=True,
                                    units=base_units,
                                    shape=shape,
                                    targets={
                                        phase_name: [mission_var_name] for phase_name in base_phases
                                    },
                                )

                                self.connect(
                                    f'pre_mission.{bus_variable}',
                                    'traj.parameters:' + mission_var_name,
                                )

                    if 'post_mission_name' in variable_data:
                        # check if post_mission_variable_name is a list
                        post_mission_variable_name = variable_data['post_mission_name']
                        if not isinstance(post_mission_variable_name, list):
                            post_mission_variable_name = [post_mission_variable_name]

                        for post_mission_var_name in post_mission_variable_name:
                            self.connect(
                                f'pre_mission.{bus_variable}',
                                post_mission_var_name,
                            )

    def _connect_mission_bus_variables(self):
        all_subsystems = self.get_all_subsystems()

        # Loop through all external subsystems.
        for external_subsystem in all_subsystems:
            for phase_name, var_mapping in external_subsystem.get_post_mission_bus_variables(
                aviary_inputs=self.aviary_inputs, phase_info=self.phase_info
            ).items():
                for mission_variable_name, post_mission_variable_names in var_mapping.items():
                    if not isinstance(post_mission_variable_names, list):
                        post_mission_variable_names = [post_mission_variable_names]

                    for post_mission_var_name in post_mission_variable_names:
                        # Remove possible prefix before a `.`, like <external_subsystem_name>.<var_name>"
                        mvn_basename = mission_variable_name.rpartition('.')[-1]
                        src_name = f'traj.{phase_name}.mission_bus_variables.{mvn_basename}'
                        self.connect(src_name, post_mission_var_name)

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

        # add the engine builder `get_design_vars` dict to a collected dict from
        # the external subsystems

        # TODO : maybe in the most general case we need to handle DVs in the mission and
        # post-mission as well. For right now we just handle pre_mission
        all_subsystems = self.get_all_subsystems()

        # loop through all_subsystems and call `get_design_vars` on each subsystem
        for subsystem in all_subsystems:
            dv_dict = subsystem.get_design_vars()
            for dv_name, dv_dict in dv_dict.items():
                self.add_design_var(dv_name, **dv_dict)

        if self.mission_method is SOLVED_2DOF:
            optimize_mass = self.pre_mission_info.get('optimize_mass')
            if optimize_mass:
                self.add_design_var(
                    Mission.Design.GROSS_MASS,
                    units='lbm',
                    lower=10,
                    upper=900.0e3,
                    ref=175.0e3,
                )

        elif self.mission_method in (HEIGHT_ENERGY, TWO_DEGREES_OF_FREEDOM):
            # vehicle sizing problem
            # size the vehicle (via design GTOW) to meet a target range using all fuel
            # capacity
            if self.problem_type is ProblemType.SIZING:
                self.add_design_var(
                    Mission.Design.GROSS_MASS,
                    lower=10.0,
                    upper=None,
                    units='lbm',
                    ref=175e3,
                )
                self.add_design_var(
                    Mission.Summary.GROSS_MASS,
                    lower=10.0,
                    upper=None,
                    units='lbm',
                    ref=175e3,
                )

                self.add_subsystem(
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
                    self.add_constraint(Mission.Constraints.RANGE_RESIDUAL, equals=0, ref=10)

            # target range problem
            # fixed vehicle (design GTOW) but variable actual GTOW for off-design
            # mission range
            elif self.problem_type is ProblemType.ALTERNATE:
                self.add_design_var(
                    Mission.Summary.GROSS_MASS,
                    lower=10.0,
                    upper=900e3,
                    units='lbm',
                    ref=175e3,
                )

                self.add_constraint(Mission.Constraints.RANGE_RESIDUAL, equals=0, ref=10)

            elif self.problem_type is ProblemType.FALLOUT:
                print('No design variables for Fallout missions')

            elif self.problem_type is ProblemType.MULTI_MISSION:
                self.add_design_var(
                    Mission.Summary.GROSS_MASS,
                    lower=10.0,
                    upper=900e3,
                    units='lbm',
                    ref=175e3,
                )

                self.add_constraint(Mission.Constraints.RANGE_RESIDUAL, equals=0, ref=10)

                # We must ensure that design.gross_mass is greater than
                # mission.summary.gross_mass and this must hold true for each of the
                # different missions that is flown the result will be the
                # design.gross_mass should be equal to the mission.summary.gross_mass
                # of the heaviest mission
                self.add_subsystem(
                    'GROSS_MASS_constraint',
                    om.ExecComp(
                        'gross_mass_resid = design_mass - actual_mass',
                        design_mass={'val': 1, 'units': 'kg'},
                        actual_mass={'val': 0, 'units': 'kg'},
                        gross_mass_resid={'val': 30, 'units': 'kg'},
                    ),
                    promotes_inputs=[
                        ('design_mass', Mission.Design.GROSS_MASS),
                        ('actual_mass', Mission.Summary.GROSS_MASS),
                    ],
                    promotes_outputs=['gross_mass_resid'],
                )

                self.add_constraint('gross_mass_resid', lower=0)

            if self.mission_method is TWO_DEGREES_OF_FREEDOM:
                # problem formulation to make the trajectory work
                self.add_design_var(Mission.Takeoff.ASCENT_T_INITIAL, lower=0, upper=100, ref=30.0)
                self.add_design_var(Mission.Takeoff.ASCENT_DURATION, lower=1, upper=1000, ref=10.0)
                self.add_design_var('tau_gear', lower=0.01, upper=1.0, units='unitless', ref=1)
                self.add_design_var('tau_flaps', lower=0.01, upper=1.0, units='unitless', ref=1)
                self.add_constraint('h_fit.h_init_gear', equals=50.0, units='ft', ref=50.0)
                self.add_constraint('h_fit.h_init_flaps', equals=400.0, units='ft', ref=400.0)

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

        target_prob = self
        if parent_prob is not None and parent_prefix != '':
            target_prob = parent_prob

        traj = self.traj

        # Determine which phases to loop over, fetching them from the trajectory
        phase_items = traj._phases.items()

        # Loop over each phase and set initial guesses for the state and control
        # variables
        for idx, (phase_name, phase) in enumerate(phase_items):
            # TODO: This will be uncommented when an openmdao bug is fixed.
            # We are using a workaround for now.
            # if not phase._is_local:
            #     # Don't set anything if phase is not on this proc.
            #     continue

            if self.mission_method is SOLVED_2DOF:
                self.phase_objects[idx].apply_initial_guesses(self, 'traj', phase)

                if self.phase_info[phase_name]['user_options'].get('ground_roll') and idx == 0:
                    continue

            # If not, fetch the initial guesses specific to the phase
            # check if guesses exist for this phase
            if 'initial_guesses' in self.phase_info[phase_name]:
                guesses = self.phase_info[phase_name]['initial_guesses']
            else:
                guesses = {}

            # Add subsystem guesses
            self._add_subsystem_guesses(phase_name, phase, target_prob, parent_prefix)

            # Set initial guesses for states, controls and time for each phase.
            self.configurator.set_phase_initial_guesses(
                self, phase_name, phase, guesses, target_prob, parent_prefix
            )

    def get_all_subsystems(self, external_subsystems=None):
        all_subsystems = []
        if external_subsystems is None:
            all_subsystems.extend(self.pre_mission_info['external_subsystems'])
        else:
            all_subsystems.extend(external_subsystems)

        all_subsystems.append(self.core_subsystems['aerodynamics'])
        all_subsystems.append(self.core_subsystems['propulsion'])

        return all_subsystems

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
        all_subsystems = self.get_all_subsystems(self.phase_info[phase_name]['external_subsystems'])

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
                # val['val'] = self.process_guess_var(val['val'], key, phase)
                val['val'] = process_guess_var(val['val'], key, phase)

                # Set the initial guess in the problem
                target_prob.set_val(parent_prefix + f'traj.{phase_name}.{path_string}:{key}', **val)

    def add_fuel_reserve_component(
        self, post_mission=True, reserves_name=Mission.Design.RESERVE_FUEL
    ):
        if post_mission:
            reserve_calc_location = self.post_mission
        else:
            reserve_calc_location = self.model

        RESERVE_FUEL_FRACTION = self.aviary_inputs.get_val(
            Aircraft.Design.RESERVE_FUEL_FRACTION, units='unitless'
        )
        if RESERVE_FUEL_FRACTION != 0:
            reserve_fuel_frac = om.ExecComp(
                'reserve_fuel_frac_mass = reserve_fuel_fraction * (takeoff_mass - final_mass)',
                reserve_fuel_frac_mass={'units': 'lbm'},
                reserve_fuel_fraction={
                    'units': 'unitless',
                    'val': RESERVE_FUEL_FRACTION,
                },
                final_mass={'units': 'lbm'},
                takeoff_mass={'units': 'lbm'},
            )

            reserve_calc_location.add_subsystem(
                'reserve_fuel_frac',
                reserve_fuel_frac,
                promotes_inputs=[
                    ('takeoff_mass', Mission.Summary.GROSS_MASS),
                    ('final_mass', Mission.Landing.TOUCHDOWN_MASS),
                    ('reserve_fuel_fraction', Aircraft.Design.RESERVE_FUEL_FRACTION),
                ],
                promotes_outputs=['reserve_fuel_frac_mass'],
            )

        RESERVE_FUEL_ADDITIONAL = self.aviary_inputs.get_val(
            Aircraft.Design.RESERVE_FUEL_ADDITIONAL, units='lbm'
        )
        reserve_fuel = om.ExecComp(
            'reserve_fuel = reserve_fuel_frac_mass + reserve_fuel_additional + reserve_fuel_burned',
            reserve_fuel={'units': 'lbm', 'shape': 1},
            reserve_fuel_frac_mass={'units': 'lbm', 'val': 0},
            reserve_fuel_additional={'units': 'lbm', 'val': RESERVE_FUEL_ADDITIONAL},
            reserve_fuel_burned={'units': 'lbm', 'val': 0},
        )

        reserve_calc_location.add_subsystem(
            'reserve_fuel',
            reserve_fuel,
            promotes_inputs=[
                'reserve_fuel_frac_mass',
                ('reserve_fuel_additional', Aircraft.Design.RESERVE_FUEL_ADDITIONAL),
                ('reserve_fuel_burned', Mission.Summary.RESERVE_FUEL_BURNED),
            ],
            promotes_outputs=[('reserve_fuel', reserves_name)],
        )
