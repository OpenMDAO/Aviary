import inspect
import sys
import warnings
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import dymos as dm
import openmdao.api as om
from dymos.utils.misc import _unspecified
from openmdao.utils.mpi import MPI

from aviary.core.post_mission_group import PostMissionGroup
from aviary.core.pre_mission_group import PreMissionGroup
from aviary.interface.utils import set_warning_format
from aviary.mission.energy_state_problem_configurator import EnergyStateProblemConfigurator
from aviary.mission.solved_two_dof_problem_configurator import SolvedTwoDOFProblemConfigurator
from aviary.mission.two_dof_problem_configurator import TwoDOFProblemConfigurator
from aviary.mission.utils import get_phase_mission_bus_lengths, process_guess_var
from aviary.subsystems.aerodynamics.aerodynamics_builder import CoreAerodynamicsBuilder
from aviary.subsystems.geometry.geometry_builder import CoreGeometryBuilder
from aviary.subsystems.mass.mass_builder import CoreMassBuilder
from aviary.subsystems.performance.performance_builder import CorePerformanceBuilder
from aviary.subsystems.premission import CorePreMission
from aviary.subsystems.propulsion.engine_model import EngineModel
from aviary.subsystems.propulsion.propulsion_builder import CorePropulsionBuilder
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.subsystems.subsystem_builder import SubsystemBuilder
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.functions import get_path
from aviary.utils.merge_variable_metadata import merge_meta_data
from aviary.utils.preprocessors import preprocess_options
from aviary.utils.process_input_decks import (
    create_vehicle,
    initialization_guessing,
    update_GASP_options,
)
from aviary.utils.utils import wrapped_convert_units
from aviary.variable_info.enums import (
    EquationsOfMotion,
    LegacyCode,
    PhaseType,
    ProblemType,
    Verbosity,
)
from aviary.variable_info.functions import setup_trajectory_params
from aviary.variable_info.variables import Aircraft, Mission, Settings

TWO_DEGREES_OF_FREEDOM = EquationsOfMotion.TWO_DEGREES_OF_FREEDOM
ENERGY_STATE = EquationsOfMotion.ENERGY_STATE
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

        self.post_mission = PostMissionGroup()
        self.verbosity = verbosity
        self.external_subsystems = []
        self.engine_models = []
        self.regular_phases = []
        self.reserve_phases = []
        self.subsystems = []

        self.aviary_inputs = None
        self.meta_data = None
        self.mission_info = None

    def configure(self):
        """Configure the Aviary group."""
        aviary_options = self.aviary_inputs
        aviary_metadata = self.meta_data

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

        # Find all variables that are shape_by_conn so we don't set their shape with a stale value
        # from the default metadata. We can only find these on the next level down because
        # aviary_group's setup is not complete until after configure.
        sbc_vars = []
        for sub in self.system_iter(recurse=False, typ=om.Group):
            pr2abs = sub._resolver.prom2abs_iter('input')
            sub_inputs = [
                (k, v[0]) for k, v in pr2abs if k.startswith('aircraft') or k.startswith('mission')
            ]
            abs2meta = sub._var_abs2meta['input']

            for data in sub_inputs:
                prom_name, abs_name = data
                meta = abs2meta[abs_name]
                if meta.get('shape_by_conn') is True:
                    sbc_vars.append(prom_name)

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

            kwargs = {'units': units}
            if key not in sbc_vars:
                # Default val if var doesn't use shape_by_conn.
                kwargs['val'] = val

            self.set_input_defaults(key, **kwargs)

        # try to get all the possible EOMs from the Enums rather than specifically calling the names here
        # This will require some modifications to the enums
        mission_method = aviary_options.get_val(Settings.EQUATIONS_OF_MOTION)

        # Temporarily add extra stuff here, probably patched soon
        # add a check for traj using hasattr for pre-mission tests.
        if mission_method is ENERGY_STATE and hasattr(self, 'traj'):
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
                if self.mission_info[phase.name]['user_options'].get('distance_solve_segments'):
                    continue

                phase.nonlinear_solver = om.NonlinearRunOnce()
                phase.linear_solver = om.LinearRunOnce()
                if hasattr(phase, 'indep_states') and isinstance(
                    phase.indep_states, om.ImplicitComponent
                ):
                    phase.indep_states.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
                    phase.indep_states.linear_solver = om.DirectSolver(rhs_checking=True)

    def load_inputs(
        self,
        aircraft_data,
        phase_info=None,
        problem_configurator=None,
        phase_info_modifier=None,
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
        # `self.verbosity` is "true" verbosity for entire run. `verbosity` is verbosity
        # override for just this method
        if verbosity is not None:
            # compatibility with being passed int for verbosity
            verbosity = Verbosity(verbosity)
        else:
            verbosity = self.verbosity  # defaults to BRIEF

        # validate phase info modifier function
        if phase_info_modifier is not None:
            self._validate_phase_info_modifier(phase_info_modifier)
            self.phase_info_modifier = phase_info_modifier
        else:
            self.phase_info_modifier = None

        ## LOAD INPUT FILE ###
        # Create AviaryValues object from file (or process existing AviaryValues object
        # with default values from metadata) and generate initial guesses
        aviary_inputs, self.initialization_guesses = create_vehicle(
            aircraft_data, meta_data=self.meta_data, verbosity=verbosity
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

        # Determine which problem configurator to use based on mission_method
        if mission_method is ENERGY_STATE:
            self.configurator = EnergyStateProblemConfigurator()
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
                'settings:equations_of_motion must be one of: energy_state, 2DOF, '
                'solved_2DOF, or custom'
            )

        # TODO this should be a preprocessor step if it is required here
        if mass_method is GASP or aero_method is GASP:
            aviary_inputs = update_GASP_options(aviary_inputs)

        ## LOAD PHASE_INFO ###
        # if phase info is a file, load it
        if isinstance(phase_info, str) or isinstance(phase_info, Path):
            phase_info_path = get_path(phase_info)
            spec = spec_from_file_location('phase_info_file', str(phase_info_path))
            phase_info_file = module_from_spec(spec)
            sys.modules['phase_info_file'] = phase_info_file
            spec.loader.exec_module(phase_info_file)

            phase_info = getattr(phase_info_file, 'phase_info')

        if phase_info is None:
            phase_info = self.configurator.get_default_phase_info(self)
            if verbosity > Verbosity.BRIEF:  # VERBOSE, DEBUG
                print(
                    f'Loaded default phase_info for {self.mission_method.value.lower()} equations '
                    'of motion.'
                )

        # create a new dictionary that only contains the phases from phase_info
        self.mission_info = {}

        for phase_name in phase_info:
            if phase_name not in ['pre_mission', 'post_mission']:
                self.mission_info[phase_name] = phase_info[phase_name]

        # pre_mission and post_mission are stored in their own dictionaries.
        if 'pre_mission' in phase_info:
            self.pre_mission_info = phase_info['pre_mission']
        else:
            self.pre_mission_info = {}

        if 'post_mission' in phase_info:
            self.post_mission_info = phase_info['post_mission']
        else:
            self.post_mission_info = {}

        return self.aviary_inputs, self.verbosity

    def load_external_subsystems(self, external_subsystems: list = [], verbosity=None):
        """
        Add external subsystems to the AviaryGroup.

        Parameters
        ----------
        external_subsystems : list of SubsystemBuilders
            List of all external subsystems to be added.

        verbosity : int, Verbosity (optional)
            Sets the printout level for the entire off-design problem that is ran.
        """
        # `self.verbosity` is "true" verbosity for entire run. `verbosity` is verbosity
        # override for just this method
        if verbosity is not None:
            # compatibility with being passed int for verbosity
            verbosity = Verbosity(verbosity)
        else:
            verbosity = self.verbosity  # defaults to BRIEF

        for subsystem in external_subsystems:
            if not isinstance(subsystem, SubsystemBuilder) and verbosity >= verbosity.BRIEF:
                warnings.warn(
                    'Provided external subsystem is not a SubsystemBuilder object and will not be '
                    'loaded.'
                )
            else:
                if isinstance(subsystem, EngineModel):
                    self.engine_models.append(subsystem)
                else:
                    self.external_subsystems.append(subsystem)
                meta_data = subsystem.meta_data.copy()
                self.meta_data = merge_meta_data([self.meta_data, meta_data])

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
        for phase_name, phase in self.mission_info.items():
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
        for phase_name, phase in self.mission_info.items():
            if 'user_options' in phase:
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

        if self.engine_models == []:
            self.engine_models = [build_engine_deck(aviary_inputs)]

        for external_subsystem in self.external_subsystems:
            aviary_inputs = external_subsystem.preprocess_inputs(aviary_inputs)

        # PREPROCESSORS #
        preprocess_options(
            aviary_inputs,
            engine_models=self.engine_models,
            verbosity=verbosity,
            metadata=self.meta_data,
        )

        self.initialization_guesses = initialization_guessing(
            self.aviary_inputs, self.initialization_guesses, self.engine_models
        )

        # This function sets all the following defaults if they were not already set:
        # self.pre_mission_info, self_post_mission_info,
        # self.require_range_residual, self.target_range
        # Other specific self.*** are defined in here as well that are specific to each builder
        self.configurator.initial_guesses(self)

        # TODO this seems like the wrong place to define the core subsystems. Maybe move to
        # load_inputs?
        ## Set Up Core Subsystems ##
        perf = CorePerformanceBuilder('performance')
        prop = CorePropulsionBuilder('propulsion', engine_models=self.engine_models)
        mass = CoreMassBuilder('mass', code_origin=self.mass_method)

        # If all phases ask for tabular aero, we can skip pre-mission. Check phase_info
        tabular = False
        for phase in self.mission_info:
            if phase not in ('pre_mission', 'post_mission'):
                try:
                    if (
                        'tabular'
                        in self.mission_info[phase]['subsystem_options']['aerodynamics']['method']
                    ):
                        tabular = True
                except KeyError:
                    tabular = False

        aero = CoreAerodynamicsBuilder(
            'aerodynamics', code_origin=self.aero_method, tabular=tabular
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
            'geometry',
            code_origin=geom_code_origin,
            code_origin_to_prioritize=code_origin_to_prioritize,
        )

        subsystems = self.subsystems = [prop, geom, mass, aero, perf]
        subsystems.extend(self.external_subsystems)

        self.ode_args = {
            'aviary_options': aviary_inputs,
            'subsystems': subsystems,
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
        self.regular_phases = []
        for idx, phase_name in enumerate(self.mission_info):
            if 'user_options' in self.mission_info[phase_name]:
                if 'reserve' in self.mission_info[phase_name]['user_options']:
                    if self.mission_info[phase_name]['user_options']['reserve'] is False:
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
        Add pre-mission systems to the Aviary group. These systems are executed before the mission.

        Depending on the mission model specified (`FLOPS` or `GASP`), this method adds various
        subsystems to the aircraft model. For the `FLOPS` mission model, a takeoff phase is added
        using the Takeoff class with the number of engines and airport altitude specified. For the
        `GASP` mission model, three subsystems are added: a TaxiSegment subsystem, an ExecComp to
        calculate the time to initiate gear and flaps, and an ExecComp to calculate the speed at
        which to initiate rotation. All subsystems are promoted with aircraft and mission inputs and
        outputs as appropriate.

        A user can override this method with their own pre-mission systems as desired.
        """
        pre_mission = PreMissionGroup()
        all_subsystem_options = self.pre_mission_info.get('subsystem_options', {})

        self.add_subsystem(
            'pre_mission',
            pre_mission,
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=['aircraft:*', 'mission:*'],
        )

        # TODO temporary until way to merge PreMissionGroup and CorePreMission group is found
        core_subsystems = self.subsystems[0:5]

        # Propulsion isn't included in core pre-mission group to avoid override step in
        # configure() - instead add it now
        pre_mission.add_subsystem(
            'propulsion',
            core_subsystems[0].build_pre_mission(
                self.aviary_inputs,
                subsystem_options=all_subsystem_options.get('propulsion', {}),
            ),
        )

        default_subsystems = core_subsystems[1:5]

        pre_mission.add_subsystem(
            'core_subsystems',
            CorePreMission(
                aviary_options=self.aviary_inputs,
                subsystems=default_subsystems,
                subsystem_options=all_subsystem_options,
                process_overrides=False,
            ),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        for subsystem in self.external_subsystems:
            name = subsystem.name
            subsystem_options = all_subsystem_options.get(name, {})

            subsystem_premission = subsystem.build_pre_mission(
                self.aviary_inputs, subsystem_options=subsystem_options
            )

            if subsystem_premission is not None:
                self.pre_mission.add_subsystem(name, subsystem_premission)

        self._add_premission_external_subsystem_masses()

        if 'linear_solver' in self.pre_mission_info:
            pre_mission.linear_solver = self.pre_mission_info['linear_solver']

        if 'nonlinear_solver' in self.pre_mission_info:
            pre_mission.nonlinear_solver = self.pre_mission_info['nonlinear_solver']

        if self.pre_mission_info['include_takeoff']:
            self.configurator.add_takeoff_systems(self)

        # Calculate Mission.TOTAL_FUEL
        pre_mission.add_subsystem(
            'total_fuel_mass_comp',
            om.ExecComp(
                'total_fuel_mass = gross_mass - zero_fuel_mass',
                total_fuel_mass={'units': 'lbm'},
                gross_mass={'units': 'lbm'},
                zero_fuel_mass={'units': 'lbm'},
            ),
            promotes_inputs=[
                ('gross_mass', Mission.GROSS_MASS),
                ('zero_fuel_mass', Mission.ZERO_FUEL_MASS),
            ],
            promotes_outputs=[('total_fuel_mass', Mission.TOTAL_FUEL)],
        )

    def _add_premission_external_subsystem_masses(self):
        """
        This private method adds a mass component that captures external subsystem masses for use in
        mass buildups. The method collects the mass names of the added subsystems. This expression
        is then used to define an ExecComp (a component that evaluates a simple equation given input
        values).

        The method promotes the input and output of this ExecComp to the top level of the
        pre-mission object, allowing this calculated subsystem mass to be accessed directly from the
        pre-mission object.
        """
        mass_names = []
        # Loop through all the phases in this subsystem.
        for external_subsystem in self.external_subsystems:
            # Get all the subsystem builders for this phase.
            mass_names.extend(external_subsystem.get_mass_names(aviary_inputs=self.aviary_inputs))

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
        phase_options = self.mission_info[phase_name]

        subsystems = self.subsystems

        phase_builder = self.configurator.get_phase_builder(self, phase_name, phase_options)

        phase_object = phase_builder.from_phase_info(
            phase_name,
            phase_options,
            subsystems,
            meta_data=self.meta_data,
        )

        phase = phase_object.build_phase(aviary_options=self.aviary_inputs)

        self.phase_objects.append(phase_object)

        # This fills in all defaults from the phase_builders user_options.
        full_options = phase_object.user_options.to_phase_info()
        self.mission_info[phase_name]['user_options'] = full_options

        # TODO: Should some of this stuff be moved into the phase builder?
        self.configurator.set_phase_options(self, phase_name, phase_idx, phase, full_options, comm)

        return phase

    def add_phases(self, parallel_phases=True, verbosity=None, comm=None):
        """
        Add the mission phases to the problem trajectory based on the user-specified
        phase_info dictionary.

        Parameters
        ----------
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

        if self.phase_info_modifier is not None:
            self.mission_info, self.post_mission_info = self.phase_info_modifier(
                self.mission_info, self.post_mission_info, self.aviary_inputs
            )

        mission_info = self.mission_info

        traj = self.add_subsystem('traj', dm.Trajectory(parallel_phases=parallel_phases))

        self.phase_objects = []

        # Get all post_mission bus vars once.
        # TODO: This method returns a dictionary keyed by phase name, but our
        # philosophy is moving away from this.
        all_subsystems = self.subsystems
        mbvars_by_sys = {}
        for subsystem in all_subsystems:
            mbvars_by_sys[subsystem.name] = subsystem.get_post_mission_bus_variables(
                self.aviary_inputs,
                mission_info=mission_info,
            )

        # Process all subsystems for all phases.
        external_parameters = {}
        for phase_idx, phase_name in enumerate(mission_info):
            # Create and add phases.
            # This also expands mission_info to include all keys.
            phase = traj.add_phase(phase_name, self._get_phase(phase_name, phase_idx, comm))

            phase_info = mission_info[phase_name]
            external_parameters[phase_name] = {}
            user_options = phase_info.get('user_options', {})
            all_subsystem_options = phase_info.get('subsystem_options', {})

            for subsystem in all_subsystems:
                if subsystem.name in all_subsystem_options:
                    subsystem_options = all_subsystem_options[subsystem.name]
                else:
                    subsystem_options = {}

                # Get all parameters and assemble them.
                parameter_dict = subsystem.get_parameters(
                    aviary_inputs=self.aviary_inputs,
                    user_options=user_options,
                    subsystem_options=subsystem_options,
                )
                # We can't guarantee a consistent order from user-provided dicts, so sort params.
                for parameter in sorted(parameter_dict):
                    external_parameters[phase_name][parameter] = parameter_dict[parameter]

                # Get all timeseries outputs and add them.
                timeseries_to_add = subsystem.get_timeseries(
                    aviary_inputs=self.aviary_inputs,
                    user_options=user_options,
                    subsystem_options=subsystem_options,
                )
                for timeseries in timeseries_to_add:
                    phase.add_timeseries_output(timeseries)

                # Add bus variables to this phase.
                mbvars = mbvars_by_sys[subsystem.name]
                if mbvars:
                    mbvars_this_phase = mbvars.get(phase_name, {})
                    for timeseries in mbvars_this_phase:
                        phase.add_timeseries_output(timeseries, timeseries='mission_bus_variables')

        traj = setup_trajectory_params(
            self,
            traj,
            self.aviary_inputs,
            list(mission_info.keys()),
            meta_data=self.meta_data,
            external_parameters=external_parameters,
        )

        self.traj = traj

        return traj

    def add_post_mission_systems(self, verbosity=None):
        """
        Add post-mission systems to the aircraft model. This is akin to the pre-mission group or the
        "premission_systems", but occurs after the mission in the execution order.

        Depending on the mission model specified (`FLOPS` or `GASP`), this method adds various
        subsystems to the aircraft model. For the `FLOPS` mission model, a landing phase is added
        using the Landing class with the wing area and lift coefficient specified, and a takeoff
        constraints ExecComp is added to enforce mass, range, velocity, and altitude continuity
        between the takeoff and climb phases. The landing subsystem is promoted with aircraft and
        mission inputs and outputs as appropriate, while the takeoff constraints ExecComp is only
        promoted with mission inputs and outputs.

        For the `GASP` mission model, four subsystems are added: a LandingSegment subsystem, an
        ExecComp to calculate the reserve fuel required, an ExecComp to calculate the overall fuel
        burn, and three ExecComps to calculate various mission objectives and constraints. All
        subsystems are promoted with aircraft and mission inputs and outputs as appropriate.

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

        # Make dymos state outputs easy to access later
        self.add_subsystem(
            'state_output',
            om.ExecComp(
                ['mass_final = mass_in', 'time_final = time_in', 'range_final = range_in'],
                mass_in={'units': 'lbm'},
                mass_final={'units': 'lbm'},
                time_in={'units': 'min'},
                time_final={'units': 'min'},
                range_in={'units': 'nmi'},
                range_final={'units': 'nmi'},
            ),
            promotes_outputs={
                ('mass_final', Mission.FINAL_MASS),
                ('time_final', Mission.FINAL_TIME),
                ('range_final', Mission.RANGE),
            },
        )

        self.configurator.add_post_mission_systems(self)

        # Add all post-mission subsystems.
        all_subsystem_options = self.pre_mission_info.get('subsystem_options', {})
        phase_mission_bus_lengths = get_phase_mission_bus_lengths(self.traj)
        for subsystem in self.subsystems:
            name = subsystem.name
            subsystem_options = all_subsystem_options.get(name, {})

            subsystem_postmission = subsystem.build_post_mission(
                aviary_inputs=self.aviary_inputs,
                mission_info=self.mission_info,
                subsystem_options=subsystem_options,
                phase_mission_bus_lengths=phase_mission_bus_lengths,
            )

            if subsystem_postmission is not None:
                post_mission.add_subsystem(name, subsystem_postmission)

        # Check if regular_phases[] is accessible
        try:
            self.regular_phases[0]
        except BaseException:
            raise ValueError(
                'regular_phases[] dictionary is not accessible. For ENERGY_STATE and '
                'SOLVED_2DOF missions, check_and_preprocess_inputs() must be called '
                'before add_post_mission_systems().'
            )

        # Fuel burn in taxi + takeoff + regular phases
        post_mission.add_subsystem(
            'fuel_burned',
            om.ExecComp(
                'fuel_burned = initial_mass - mass_final',
                initial_mass={'units': 'lbm'},
                mass_final={
                    'units': 'lbm'
                },  # this final mass already includes fuel burned in taxi and takeoff
                fuel_burned={'units': 'lbm'},
            ),
            promotes_inputs=[('initial_mass', Mission.GROSS_MASS)],
            promotes_outputs=[('fuel_burned', Mission.FUEL)],
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
                promotes=[('reserve_fuel_burned', Mission.RESERVE_FUEL)],
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

        # Ensure that the usable fuel loaded onto the aircraft is greater or equal to the mission fuel + reserve fuel
        # The aircraft will naturally try to mimize 'total_fuel_mass_constraint' so it's not carrying extra unnecessary fuel
        post_mission.add_subsystem(
            'total_fuel_mass_con',
            om.ExecComp(
                'total_fuel_mass_constraint = total_fuel_mass - mission_fuel_burned - reserve_fuel',
                total_fuel_mass_constraint={'units': 'lbm'},
                total_fuel_mass={'units': 'lbm'},
                mission_fuel_burned={'units': 'lbm'},
                reserve_fuel={'units': 'lbm'},
            ),
            promotes_inputs=[
                ('total_fuel_mass', Mission.TOTAL_FUEL),
                ('mission_fuel_burned', Mission.FUEL),
                ('reserve_fuel', Mission.TOTAL_RESERVE_FUEL),
            ],
            promotes_outputs=[('total_fuel_mass_constraint', Mission.Constraints.MASS_RESIDUAL)],
        )
        # Users can set the below constraint to lower=0.0, which will allow for more fuel on the aircraft than the mission
        # requires. however, caution will need to be taken to ensure the ref is of the right magnitude otherwise the optimizer
        # may not try as hard as needed to minimize this.
        if Settings.EQUATIONS_OF_MOTION is SOLVED_2DOF:
            # For missions where we are allowed to have more fuel in the tanks than we burn during the mission.
            self.add_constraint(
                Mission.Constraints.MASS_RESIDUAL,
                lower=0.0,
                ref=1e5,
            )
        else:
            self.add_constraint(
                Mission.Constraints.MASS_RESIDUAL,
                equals=0.0,
                ref=1e5,
            )

        # If a target distance (or time) has been specified for this phase distance (or time) is
        # measured from the start of this phase to the end of this phase
        for phase_name in self.mission_info:
            user_options = self.mission_info[phase_name]['user_options']

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
            integrates_mass = user_options['phase_type'] is PhaseType.BREGUET_RANGE

            if integrates_mass and time_duration is not None:
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
            'excess_fuel_capacity = total_fuel_capacity - unusable_fuel - overall_fuel',
            total_fuel_capacity={'units': 'lbm'},
            unusable_fuel={'units': 'lbm'},
            overall_fuel={'units': 'lbm'},
            excess_fuel_capacity={'units': 'lbm'},
        )

        post_mission.add_subsystem(
            'excess_fuel_constraint',
            ecomp,
            promotes_inputs=[
                ('total_fuel_capacity', Aircraft.Fuel.TOTAL_CAPACITY),
                ('unusable_fuel', Aircraft.Fuel.UNUSABLE_FUEL_MASS),
                ('overall_fuel', Mission.TOTAL_FUEL),
            ],
            promotes_outputs=[('excess_fuel_capacity', Mission.Constraints.EXCESS_FUEL_CAPACITY)],
        )

        # determine if the user wants the excess_fuel_capacity constraint active and if so add it to the problem
        if Aircraft.Fuel.IGNORE_FUEL_CAPACITY_CONSTRAINT in self.aviary_inputs:
            ignore_capacity_constraint = self.aviary_inputs.get_val(
                Aircraft.Fuel.IGNORE_FUEL_CAPACITY_CONSTRAINT, units='unitless'
            )
        else:
            ignore_capacity_constraint = self.meta_data[
                Aircraft.Fuel.IGNORE_FUEL_CAPACITY_CONSTRAINT
            ]['default_value']
            self.aviary_inputs.set_val(
                Aircraft.Fuel.IGNORE_FUEL_CAPACITY_CONSTRAINT,
                val=ignore_capacity_constraint,
                units='unitless',
            )

        if not ignore_capacity_constraint:
            self.add_constraint(
                Mission.Constraints.EXCESS_FUEL_CAPACITY, lower=0, ref=1.0e5, units='lbm'
            )
        else:
            if verbosity >= Verbosity.BRIEF:
                warnings.warn(
                    'Aircraft.Fuel.IGNORE_FUEL_CAPACITY_CONSTRAINT = True, therefore '
                    'EXCESS_FUEL_CAPACITY constraint was not added to the Aviary problem. The '
                    'aircraft may not have enough space for fuel, so check the value of '
                    'Mission.Constraints.EXCESS_FUEL_CAPACITY for details.'
                )

        post_mission.add_subsystem(
            'block_fuel_comp',
            om.ExecComp(
                'block_fuel = mission_fuel_burned + fuel_burned_taxi_in',
                block_fuel={'units': 'lbm'},
                mission_fuel_burned={'units': 'lbm'},
                fuel_burned_taxi_in={'units': 'lbm'},
            ),
            promotes_inputs=[
                ('mission_fuel_burned', Mission.FUEL),
                ('fuel_burned_taxi_in', Mission.Taxi.FUEL_TAXI_IN),
            ],
            promotes_outputs=[('block_fuel', Mission.BLOCK_FUEL)],
        )

    def link_phases(self, verbosity=None, comm=None):
        """
        Link phases together after they've been added.

        Based on which phases the user has selected, we might need special logic to do the Dymos
        linkages correctly. Some of those connections for the simple GASP and FLOPS mission are
        shown here.
        """
        # `self.verbosity` is "true" verbosity for entire run. `verbosity` is verbosity override for
        # just this method
        if verbosity is not None:
            # compatibility with being passed int for verbosity
            verbosity = Verbosity(verbosity)
        else:
            verbosity = self.verbosity  # defaults to BRIEF

        self._add_bus_variables_and_connect()
        self._connect_mission_bus_variables()

        final_phase = self.regular_phases[-1]

        # We connect the last points in the trajectory to the state_output component to make it
        # easier for users to access Mission.FINAL_MASS, Mission.FINAL_TIME,
        # and Mission.RANGE.
        self.connect(
            f'traj.{final_phase}.states:mass',
            'state_output.mass_in',
            src_indices=[-1],
        )
        self.connect(
            f'traj.{final_phase}.timeseries.distance',
            'state_output.range_in',
            src_indices=[-1],
        )
        self.connect(
            f'traj.{final_phase}.timeseries.time', 'state_output.time_in', src_indices=[-1]
        )

        phases = list(self.mission_info.keys())

        if len(phases) <= 1:
            return

        # In summary, the following code loops over all phases in self.mission_info, gets the linked
        # variables from each external subsystem in each phase, and stores the lists of linked
        # variables in lists_to_link. It then gets a list of unique variable names from
        # lists_to_link and loops over them, creating a list of phase names for each variable and
        # linking the phases using self.traj.link_phases().

        lists_to_link = []
        for idx, phase_name in enumerate(self.mission_info):
            lists_to_link.append([])
            for external_subsystem in self.external_subsystems:
                lists_to_link[idx].extend(
                    external_subsystem.get_linked_variables(aviary_inputs=self.aviary_inputs)
                )

        # get unique variable names from lists_to_link
        unique_vars = list(set([var for sublist in lists_to_link for var in sublist]))

        # Phase linking.
        # If we are under mpi, and traj.phases is running in parallel, then let the optimizer handle
        # the linkage constraints.  Note that we can technically parallelize connected phases, but
        # it requires a solver that we would like to avoid.
        true_unless_mpi = True
        if comm.size > 1 and self.traj.options['parallel_phases']:
            true_unless_mpi = False

        # loop over unique variable names
        for var in unique_vars:
            phases_to_link = []
            for idx, phase_name in enumerate(self.mission_info):
                if var in lists_to_link[idx]:
                    phases_to_link.append(phase_name)

            if len(phases_to_link) > 1:  # TODO: hack
                # go phase by phase and either directly link if two standard phases, or use linkage
                # constraint if either are analytic
                # TODO need more unified way to handle this instead of splitting between AviaryGroup
                #      and configurators
                for ii in range(len(phases) - 1):
                    phase1, phase2 = phases[ii : ii + 2]
                    opt1 = self.mission_info[phase1]['user_options']
                    opt2 = self.mission_info[phase2]['user_options']
                    integrates_mass1 = opt1['phase_type'] is PhaseType.BREGUET_RANGE
                    integrates_mass2 = opt2['phase_type'] is PhaseType.BREGUET_RANGE

                    if integrates_mass1 or integrates_mass2:
                        # TODO need ref value for these linkage constraints
                        self.traj.add_linkage_constraint(phase1, phase2, var, var, connected=False)
                    else:
                        self.traj.link_phases(phases=[phase1, phase2], vars=[var], connected=True)

        self.configurator.link_phases(self, phases, connect_directly=true_unless_mpi)

        self.configurator.check_trajectory(self)

    def _add_bus_variables_and_connect(self):
        all_subsystems = self.subsystems

        base_phases = list(self.mission_info.keys())

        for subsystem in all_subsystems:
            bus_variables = subsystem.get_pre_mission_bus_variables(
                self.aviary_inputs, mission_info=self.mission_info
            )
            if bus_variables is not None:
                for bus_variable, variable_data in bus_variables.items():
                    if 'mission_name' in variable_data:
                        mission_var_names = variable_data['mission_name']
                        src_indices = variable_data.get('src_indices', None)

                        # check if mission_variable_name is a list
                        if not isinstance(mission_var_names, list):
                            mission_var_names = [mission_var_names]

                        # loop over the mission_variable_name list and add each variable to
                        # the trajectory
                        for mission_var_name in mission_var_names:
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
                                            src_indices=src_indices,
                                        )

                                else:
                                    self.traj.add_parameter(
                                        mission_var_name,
                                        opt=False,
                                        static_target=True,
                                        units=base_units,
                                        shape=shape,
                                        targets={
                                            phase_name: [targets] for phase_name in base_phases
                                        },
                                    )

                                    self.connect(
                                        f'pre_mission.{bus_variable}',
                                        'traj.parameters:' + mission_var_name,
                                        src_indices=src_indices,
                                    )

                    if 'post_mission_name' in variable_data:
                        # check if post_mission_variable_name is a list
                        post_mission_var_names = variable_data['post_mission_name']
                        src_indices = variable_data.get('src_indices', None)

                        if not isinstance(post_mission_var_names, list):
                            post_mission_var_names = [post_mission_var_names]

                        for post_mission_var_name in post_mission_var_names:
                            self.connect(
                                f'pre_mission.{bus_variable}',
                                f'{post_mission_var_name}',
                                src_indices=src_indices,
                            )

    def _connect_mission_bus_variables(self):
        all_subsystems = self.subsystems

        # Loop through all external subsystems.
        for subsystem in all_subsystems:
            for phase_name, var_mapping in subsystem.get_post_mission_bus_variables(
                aviary_inputs=self.aviary_inputs, mission_info=self.mission_info
            ).items():
                for mission_variable_name, variable_data in var_mapping.items():
                    post_mission_variable_names = variable_data['post_mission_name']
                    src_indices = variable_data.get('src_indices', None)
                    if not isinstance(post_mission_variable_names, list):
                        post_mission_variable_names = [post_mission_variable_names]

                    for post_mission_var_name in post_mission_variable_names:
                        # Remove possible prefix before a `.`, like <external_subsystem_name>.<var_name>"
                        mvn_basename = mission_variable_name.rpartition('.')[-1]
                        src_name = f'traj.{phase_name}.mission_bus_variables.{mvn_basename}'
                        self.connect(src_name, post_mission_var_name, src_indices=src_indices)

    def add_design_variables(self, problem_type: ProblemType = None, verbosity=None):
        """
        Adds design variables to the Aviary problem.

        Depending on the mission model and problem type, different design variables and constraints
        are added.

        If using the FLOPS model, a design variable is added for the gross mass of the aircraft,
        with a lower bound of 10 lbm and an upper bound of 900,000 lbm.

        If using the GASP model, the following design variables are added depending on the mission
        type:
        - the initial thrust-to-weight ratio of the aircraft during ascent
        - the duration of the ascent phase
        - the time constant for the landing gear actuation
        - the time constant for the flaps actuation

        In addition, two constraints are added for the GASP model:
        - the initial altitude of the aircraft with gear extended is constrained to be 50 ft
        - the initial altitude of the aircraft with flaps extended is constrained to be 400 ft

        If solving a sizing problem, a design variable is added for the gross mass of the aircraft,
        and another for the gross mass of the aircraft computed during the mission. A constraint is
        also added to ensure that the residual range is zero.

        If solving an OFF_DESIGN_MIN_FUEL problem, only a design variable for the gross mass of the aircraft
        computed during the mission is added. A constraint is also added to ensure that the residual
        range is zero.

        In all cases, a design variable is added for the final cruise mass of the aircraft, with no
        upper bound, and a residual mass constraint is added to ensure that the mass balances.
        """
        # `self.verbosity` is "true" verbosity for entire run. `verbosity` is verbosity
        # override for just this method
        if verbosity is not None:
            # compatibility with being passed int for verbosity
            verbosity = Verbosity(verbosity)
        else:
            verbosity = self.verbosity  # defaults to BRIEF

        all_subsystems = self.subsystems

        # loop through all_subsystems and call `get_design_vars` on each subsystem
        for subsystem in all_subsystems:
            dv_dict = subsystem.get_design_vars(aviary_inputs=self.aviary_inputs)
            for dv_name, dv_dict in dv_dict.items():
                self.add_design_var(dv_name, **dv_dict)

        if self.mission_method is SOLVED_2DOF:  # TODO: to be removed soon
            optimize_mass = self.pre_mission_info.get('optimize_mass')
            if optimize_mass:
                self.add_design_var(
                    Aircraft.Design.GROSS_MASS,
                    units='lbm',
                    lower=10,
                    upper=900.0e3,
                    ref=175.0e3,
                )

        elif self.mission_method in (
            ENERGY_STATE,
            TWO_DEGREES_OF_FREEDOM,
        ):  # TODO: This becomes generic as soon as SOLVED_2DOF is removed
            # vehicle sizing problem
            # size the vehicle (via design GTOW) to meet a target range using all fuel
            # capacity
            if problem_type is ProblemType.SIZING:
                self.add_design_var(
                    Aircraft.Design.GROSS_MASS,
                    lower=10.0,
                    upper=None,
                    units='lbm',
                    ref=175e3,
                )
                self.add_design_var(
                    Mission.GROSS_MASS,
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
                        ('lhs:GTOW', Aircraft.Design.GROSS_MASS),
                        ('rhs:GTOW', Mission.GROSS_MASS),
                    ],
                )

                if self.require_range_residual:
                    self.add_constraint(Mission.Constraints.RANGE_RESIDUAL, equals=0, ref=1000)

            elif problem_type is ProblemType.OFF_DESIGN_MIN_FUEL:
                # target range problem
                # fixed vehicle (design GTOW) but variable actual GTOW for off-design
                # get the design gross mass and set as the upper bound for the gross mass design variable
                MTOW = self.aviary_inputs.get_val(Aircraft.Design.GROSS_MASS, 'lbm')
                self.add_design_var(
                    Mission.GROSS_MASS,
                    lower=10.0,
                    upper=MTOW,
                    units='lbm',
                    ref=MTOW,
                )

                self.add_constraint(Mission.Constraints.RANGE_RESIDUAL, equals=0, ref=1000)

            elif problem_type is ProblemType.OFF_DESIGN_MAX_RANGE:
                # fixed vehicle gross mass aviary finds optimal trajectory and maximum range
                if verbosity >= Verbosity.VERBOSE:
                    print(
                        'No additional aircraft design variables added for OFF_DESIGN_MAX_RANGE missions'
                    )

            elif problem_type is ProblemType.MULTI_MISSION:
                self.add_design_var(
                    Mission.GROSS_MASS,
                    lower=10.0,
                    upper=900e3,
                    units='lbm',
                    ref=175e3,
                )

                # TODO: RANGE_RESIDUAL constraint should be added based on what the
                # user sets as the objective. if Objective is not range or Mission.RANGE,
                # the range constriant should be added to make target rage = summary range
                self.add_constraint(Mission.Constraints.RANGE_RESIDUAL, equals=0, ref=1000)

                # We must ensure that design.gross_mass is greater than  Mission.GROSS_MASS
                # and this must hold true for each of the different missions that is flown the
                # result will be the design.gross_mass should be equal to the
                # Mission.GROSS_MASS of the heaviest mission
                self.add_subsystem(
                    'GROSS_MASS_constraint',
                    om.ExecComp(
                        'gross_mass_resid = design_mass - actual_mass',
                        design_mass={'val': 1, 'units': 'kg'},
                        actual_mass={'val': 0, 'units': 'kg'},
                        gross_mass_resid={'val': 30, 'units': 'kg'},
                    ),
                    promotes_inputs=[
                        ('design_mass', Aircraft.Design.GROSS_MASS),
                        ('actual_mass', Mission.GROSS_MASS),
                    ],
                    promotes_outputs=['gross_mass_resid'],
                )

                # ref scales gross_mass_resid = design_mass - actual_mass to O(1).
                # For fleet missions much lighter than design, residuals can be
                # 10-20% of design mass. GROSS_MASS/4 puts scaled values ~0.2-0.6.
                _gm_ref = self.aviary_inputs.get_val(Aircraft.Design.GROSS_MASS, 'kg') / 4.0
                self.add_constraint('gross_mass_resid', lower=0, ref=_gm_ref)

            if self.mission_method is TWO_DEGREES_OF_FREEDOM:
                # TODO: This should be moved into the problem configurator b/c it's 2DOF specific
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
        # any mission that does not have any dymos phases, there is nothing to set.
        if not hasattr(self, 'traj'):
            return
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

                if self.mission_info[phase_name]['user_options'].get('ground_roll') and idx == 0:
                    continue

            # If not, fetch the initial guesses specific to the phase
            # check if guesses exist for this phase
            if 'initial_guesses' in self.mission_info[phase_name]:
                guesses = self.mission_info[phase_name]['initial_guesses'].copy()
            else:
                guesses = {}

            # Add subsystem guesses
            self._add_subsystem_guesses(phase_name, phase, target_prob, parent_prefix)

            # Set initial guesses for states, controls and time for each phase.
            self.configurator.set_phase_initial_guesses(
                self, phase_name, phase, guesses, target_prob, parent_prefix
            )

    def _add_subsystem_guesses(self, phase_name, phase, target_prob, parent_prefix):
        """
        Adds the initial guesses for each subsystem of a given phase to the problem. This method
        first fetches all subsystems associated with the given phase. It then loops over each
        subsystem and fetches its initial guesses. For each guess, it identifies whether the guess
        corresponds to a state or a control variable and then processes the guess variable. After
        this, the initial guess is set in the problem using the `set_val` method.

        Parameters
        ----------
        phase_name : str
            The name of the phase for which the subsystem guesses are being added.
        phase : Phase
            The phase object for which the subsystem guesses are being added.
        """
        all_subsystems = self.subsystems
        phase_info = self.mission_info[phase_name]
        user_options = phase_info.get('user_options', {})
        all_subsystem_options = phase_info.get('subsystem_options', {})

        # Loop over each subsystem
        for subsystem in all_subsystems:
            if subsystem.name in all_subsystem_options:
                subsystem_options = all_subsystem_options[subsystem.name]
            else:
                subsystem_options = {}

            # Fetch the initial guesses for the subsystem
            initial_guesses = subsystem.get_initial_guesses(
                aviary_inputs=self.aviary_inputs,
                user_options=user_options,
                subsystem_options=subsystem_options,
            )

            # Loop over each guess
            for key, val_dict in initial_guesses.items():
                # Identify the type of the guess (state or control)
                var_type = val_dict['type']
                if 'state' in var_type:
                    path_string = 'states'
                elif 'control' in var_type:
                    path_string = 'controls'

                # Process the guess variable (handles array interpolation)
                # val['val'] = self.process_guess_var(val['val'], key, phase)
                val = process_guess_var(val_dict['val'], key, phase)

                # Set the initial guess in the problem
                target_prob.set_val(
                    parent_prefix + f'traj.{phase_name}.{path_string}:{key}',
                    val,
                    units=val_dict.get('units', None),
                )

    def add_fuel_reserve_component(
        self, post_mission=True, reserves_name=Mission.TOTAL_RESERVE_FUEL
    ):
        if post_mission:
            reserve_calc_location = self.post_mission
        else:
            reserve_calc_location = self.model

        reserve_fuel_margin = self.aviary_inputs.get_val(
            Mission.RESERVE_FUEL_MARGIN, units='unitless'
        )
        if reserve_fuel_margin != 0:
            # Originally tried to reference Mission.FUEL for fuel burn but in some tests this led to errors
            reserve_fuel_frac = om.ExecComp(
                'reserve_fuel_margin_mass = reserve_fuel_margin / 100 * (initial_mass - final_mass)',
                reserve_fuel_margin_mass={'units': 'lbm'},
                reserve_fuel_margin={
                    'units': 'unitless',
                    'val': reserve_fuel_margin,
                },
                initial_mass={'units': 'lbm'},
                final_mass={'units': 'lbm'},
            )

            reserve_calc_location.add_subsystem(
                'reserve_fuel_frac',
                reserve_fuel_frac,
                promotes_inputs=[
                    ('initial_mass', Mission.GROSS_MASS),
                    ('reserve_fuel_margin', Mission.RESERVE_FUEL_MARGIN),
                ],
                promotes_outputs=['reserve_fuel_margin_mass'],
            )
            # connect final mass
            self.connect(
                f'traj.{self.regular_phases[-1]}.timeseries.mass',
                'reserve_fuel_frac.final_mass',
                src_indices=[-1],
            )

        reserve_fuel_additional = self.aviary_inputs.get_val(
            Mission.RESERVE_FUEL_ADDITIONAL, units='lbm'
        )
        reserve_fuel = om.ExecComp(
            'reserve_fuel = reserve_fuel_margin_mass + reserve_fuel_additional + reserve_fuel_burned',
            reserve_fuel={'units': 'lbm', 'shape': 1},
            reserve_fuel_margin_mass={'units': 'lbm', 'val': 0},
            reserve_fuel_additional={'units': 'lbm', 'val': reserve_fuel_additional},
            reserve_fuel_burned={'units': 'lbm', 'val': 0},
        )

        reserve_calc_location.add_subsystem(
            'reserve_fuel',
            reserve_fuel,
            promotes_inputs=[
                'reserve_fuel_margin_mass',
                ('reserve_fuel_additional', Mission.RESERVE_FUEL_ADDITIONAL),
                ('reserve_fuel_burned', Mission.RESERVE_FUEL),
            ],
            promotes_outputs=[('reserve_fuel', reserves_name)],
        )

    def _validate_phase_info_modifier(self, phase_info_modifier):
        """Check function for required arguments (phase_info, post_mission_info, aviary_inputs)"""

        # validate phase_info_modifier function
        sig = inspect.signature(phase_info_modifier)
        params = sig.parameters
        # NOTE Exact argument name matching needed to check types later (this might be
        #      avoidable, if params is guaranteed to be in order)
        expected_args = {'phase_info', 'post_mission_info', 'aviary_inputs'}
        actual_args = set(params.keys())

        if expected_args != actual_args:
            raise ValueError(
                f'Phase modifier function must match arguments: {expected_args}. Got: {actual_args}'
            )

        # Check argument types (if provided)
        if params['phase_info'].annotation not in (dict, inspect.Parameter.empty):
            raise TypeError(
                "The 'phase_info' argument of phase info modifier function must be a dict (or "
                'left unspecified).'
            )
        if params['post_mission_info'].annotation not in (dict, inspect.Parameter.empty):
            raise TypeError(
                "The 'post_mission_info' argument of phase info modifier function must be a dict "
                '(or left unspecified).'
            )
        if params['aviary_inputs'].annotation not in (AviaryValues, inspect.Parameter.empty):
            raise TypeError(
                "The 'aviary_inputs' argument of phase info modifier function must be an "
                'AviaryValues (or left unspecified).'
            )
