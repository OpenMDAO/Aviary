import openmdao.api as om
from openmdao.utils.mpi import MPI

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import EquationsOfMotion
from aviary.variable_info.variables import Settings
from aviary.variable_info.enums import Verbosity
from aviary.core.PreMissionGroup import PreMissionGroup

HEIGHT_ENERGY = EquationsOfMotion.HEIGHT_ENERGY


class AviaryGroup(om.Group):
    """
    A standard OpenMDAO group that handles Aviary's promotions in the configure
    method. This assures that we only call set_input_defaults on variables
    that are present in the model.
    """

    def __init__(self, analysis_scheme=AnalysisScheme.COLLOCATION, verbosity=None, **kwargs):
        super().__init__(**kwargs)

        self.pre_mission = PreMissionGroup()

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
        self.builder = [] # what does this do?

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
        # We haven't read the input data yet, we don't know what desired run verbosity is
        # `self.verbosity` is "true" verbosity for entire run. `verbosity` is verbosity
        # override for just this method
        if verbosity is not None:
            # compatibility with being passed int for verbosity
            verbosity = Verbosity(verbosity)
        else:
            verbosity = self.verbosity  # usually None

        ## LOAD INPUT FILE ###
        # Create AviaryValues object from file (or process existing AviaryValues object
        # with default values from metadata) and generate initial guesses
        aviary_inputs, self.initialization_guesses = create_vehicle(
            aircraft_data, meta_data=meta_data, verbosity=verbosity
        )

        # update verbosity now that we have read the input data
        self.verbosity = aviary_inputs.get_val(Settings.VERBOSITY)
        # if user did not ask for verbosity override for this method, use value from data
        if verbosity is None:
            verbosity = aviary_inputs.get_val(Settings.VERBOSITY)

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
        if phase_info is None:
            # check if the user generated a phase_info from gui
            # Load the phase info dynamically from the current working directory
            phase_info_module_path = Path.cwd() / 'outputted_phase_info.py'

            if phase_info_module_path.exists():
                spec = importlib.util.spec_from_file_location(
                    'outputted_phase_info', phase_info_module_path
                )
                outputted_phase_info = importlib.util.module_from_spec(spec)
                sys.modules['outputted_phase_info'] = outputted_phase_info
                spec.loader.exec_module(outputted_phase_info)

                # Access the phase_info variable from the loaded module
                phase_info = outputted_phase_info.phase_info

                # if verbosity level is BRIEF or higher, print that we're using the
                # outputted phase info
                if verbosity >= Verbosity.BRIEF:
                    print('Using outputted phase_info from current working directory')
            else:
                phase_info = self.configurator.get_default_phase_info(self)

                if verbosity is not None and verbosity >= Verbosity.BRIEF:
                    print(
                        'Loaded default phase_info for '
                        f'{self.mission_method.value.lower()} equations of motion'
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
                if (
                    self.analysis_scheme is AnalysisScheme.COLLOCATION
                    and self.mission_method is EquationsOfMotion.TWO_DEGREES_OF_FREEDOM
                ):
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
                    time_duration = phase['user_options']['time_duration']
                    if time_duration[0] is not None and time_duration[0] <= 0:
                        raise ValueError(
                            f'Invalid time_duration in phase_info[{phase_name}]'
                            f'[user_options]. Current (value: {time_duration[0]}), '
                            f'(units: {time_duration[1]}) <= 0")'
                        )

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

        self._update_metadata_from_subsystems()
        self._check_reserve_phase_separation()

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