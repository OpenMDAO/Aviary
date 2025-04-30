import numpy as np
import openmdao.api as om
from dymos.transcriptions.transcription_base import TranscriptionBase

from aviary.mission.flight_phase_builder import FlightPhaseOptions
from aviary.mission.flops_based.phases.build_landing import Landing
from aviary.mission.flops_based.phases.build_takeoff import Takeoff
from aviary.mission.flops_based.phases.energy_phase import EnergyPhase
from aviary.mission.phase_builder_base import PhaseBuilderBase
from aviary.mission.problem_configurator import ProblemConfiguratorBase
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.process_input_decks import initialization_guessing
from aviary.utils.utils import wrapped_convert_units
from aviary.variable_info.enums import AnalysisScheme, LegacyCode
from aviary.variable_info.variables import Aircraft, Dynamic, Mission

if hasattr(TranscriptionBase, 'setup_polynomial_controls'):
    use_new_dymos_syntax = False
else:
    use_new_dymos_syntax = True


class HeightEnergyProblemConfigurator(ProblemConfiguratorBase):
    """
    A Height-Energy specific builder that customizes AviaryProblem() for use with
    height energy phases.
    """

    def check_phase_options(self, prob):
        """Returns the Options Dictionary used to instantiate the phases used by this ODE."""
        ' This will be used by check_and_preprocess_inputs in M4L2 to ensure that the '
        ' required inputs are in the phase_info.'
        return FlightPhaseOptions

    def initial_guesses(self, prob):
        """
        Set any initial guesses for variables in the aviary problem.

        This is called at the end of AivaryProblem.load_inputs.

        Parameters
        ----------
        prob : AviaryProblem
            Problem that owns this builder.
        """
        # TODO: This should probably be moved to the set_initial_guesses() method in AviaryProblem class
        # Defines how the problem should build it's initial guesses for load_inputs()
        # this modifies mass_method, initialization_guesses, and aviary_values

        aviary_inputs = prob.aviary_inputs

        if prob.engine_builders is None:
            prob.engine_builders = [build_engine_deck(aviary_inputs)]

        prob.initialization_guesses = initialization_guessing(
            aviary_inputs, prob.initialization_guesses, prob.engine_builders
        )

        # Deal with missing defaults in phase info:
        prob.pre_mission_info.setdefault('include_takeoff', True)
        prob.pre_mission_info.setdefault('external_subsystems', [])

        prob.post_mission_info.setdefault('include_landing', True)
        prob.post_mission_info.setdefault('external_subsystems', [])

        # Commonly referenced values
        aviary_inputs.set_val(
            Mission.Summary.GROSS_MASS,
            val=prob.initialization_guesses['actual_takeoff_mass'],
            units='lbm',
        )

        if 'target_range' in prob.post_mission_info:
            aviary_inputs.set_val(
                Mission.Summary.RANGE,
                wrapped_convert_units(prob.post_mission_info['target_range'], 'NM'),
                units='NM',
            )
            prob.require_range_residual = True
            prob.target_range = wrapped_convert_units(prob.post_mission_info['target_range'], 'NM')
        else:
            prob.require_range_residual = False
            # still instantiate target_range because it is used for default guesses
            # for phase comps
            prob.target_range = aviary_inputs.get_val(Mission.Design.RANGE, units='NM')

    def get_default_phase_info(self, prob):
        """
        Return a default phase_info for this type or problem.

        The default phase_info is used in the level 1 and 2 interfaces when no
        phase_info is specified.

        This is called during load_inputs.

        Parameters
        ----------
        prob : AviaryProblem
            Problem that owns this builder.

        Returns
        -------
        AviaryValues
            General default phase_info.
        """
        if prob.analysis_scheme is AnalysisScheme.COLLOCATION:
            from aviary.interface.default_phase_info.height_energy import phase_info
        else:
            raise RuntimeError('Height Energy requires that a phase_info is specified.')

        return phase_info

    def get_code_origin(self, prob):
        """
        Return the legacy of this problem configurator.

        Parameters
        ----------
        prob : AviaryProblem
            Problem that owns this builder.

        Returns
        -------
        LegacyCode
            Code origin enum.
        """
        return LegacyCode.FLOPS

    def add_takeoff_systems(self, prob):
        """
        Adds takeoff systems to the model in prob.

        Parameters
        ----------
        prob : AviaryProblem
            Problem that owns this builder.
        """
        takeoff_options = Takeoff(
            airport_altitude=0.0,  # ft
            num_engines=prob.aviary_inputs.get_val(Aircraft.Engine.NUM_ENGINES),
        )

        # Build and add takeoff subsystem
        takeoff = takeoff_options.build_phase(False)
        prob.model.add_subsystem(
            'takeoff',
            takeoff,
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=['mission:*'],
        )

    def get_phase_builder(self, prob, phase_name, phase_options):
        """
        Return a phase_builder for the requested phase.

        This is called from _get_phase in AviaryProblem.add_phases

        Parameters
        ----------
        prob : AviaryProblem
            Problem that owns this builder.
        phase_name : str
            Name of the requested phase.
        phase_options : dict
            Phase options for the requested phase.

        Returns
        -------
        PhaseBuilderBase
            Phase builder for requested phase.
        """
        if 'phase_builder' in phase_options:
            phase_builder = phase_options['phase_builder']
            if not issubclass(phase_builder, PhaseBuilderBase):
                raise TypeError(
                    'phase_builder for the phase called '
                    '{phase_name} must be a PhaseBuilderBase object.'
                )
        else:
            phase_builder = EnergyPhase

        return phase_builder

    def set_phase_options(self, prob, phase_name, phase_idx, phase, user_options):
        """
        Set any necessary problem-related options on the phase.

        This is called from _get_phase in AviaryProblem.add_phases

        Parameters
        ----------
        prob : AviaryProblem
            Problem that owns this builder.
        phase_name : str
            Name of the requested phase.
        phase_idx : int
            Phase position in prob.phases. Can be used to identify first phase.
        phase : Phase
            Instantiated phase object.
        user_options : dict
            Subdictionary "user_options" from the phase_info.
        """
        try:
            fix_initial = user_options.get_val('fix_initial')
        except KeyError:
            fix_initial = False

        try:
            fix_duration = user_options.get_val('fix_duration')
        except KeyError:
            fix_duration = False

        time_units = phase.time_options['units']

        # Make a good guess for a reasonable intitial time scaler.
        try:
            initial_bounds = user_options.get_val('initial_bounds', units=time_units)
        except KeyError:
            initial_bounds = (None, None)

        if initial_bounds[0] is not None and initial_bounds[1] != 0.0:
            # Upper bound is good for a ref.
            user_options.set_val('initial_ref', initial_bounds[1], units=time_units)
        else:
            user_options.set_val('initial_ref', 600.0, time_units)

        duration_bounds = user_options.get_val('duration_bounds', time_units)
        user_options.set_val(
            'duration_ref', (duration_bounds[0] + duration_bounds[1]) / 2.0, time_units
        )

        # The rest of the phases includes all Height Energy method phases
        # and any 2DOF phases that don't fall into the naming patterns
        # above.
        input_initial = phase_idx > 0

        if fix_initial or input_initial:
            if prob.comm.size > 1:
                # Phases are disconnected to run in parallel, so initial ref is
                # valid.
                initial_ref = user_options.get_val('initial_ref', time_units)
            else:
                # Redundant on a fixed input; raises a warning if specified.
                initial_ref = None
                initial_bounds = (None, None)

            phase.set_time_options(
                fix_initial=fix_initial,
                fix_duration=fix_duration,
                units=time_units,
                duration_bounds=user_options.get_val('duration_bounds', time_units),
                duration_ref=user_options.get_val('duration_ref', time_units),
                initial_ref=initial_ref,
            )
        else:
            phase.set_time_options(
                fix_initial=fix_initial,
                fix_duration=fix_duration,
                units=time_units,
                duration_bounds=user_options.get_val('duration_bounds', time_units),
                duration_ref=user_options.get_val('duration_ref', time_units),
                initial_bounds=initial_bounds,
                initial_ref=user_options.get_val('initial_ref', time_units),
            )

    def link_phases(self, prob, phases, connect_directly=True):
        """
        Apply any additional phase linking.

        Note that some phase variables are handled in the AviaryProblem. Only
        problem-specific ones need to be linked here.

        This is called from AviaryProblem.link_phases

        Parameters
        ----------
        prob : AviaryProblem
            Problem that owns this builder.
        phases : Phase
            Phases to be linked.
        connect_directly : bool
            When True, then connected=True. This allows the connections to be
            handled by constraints if `phases` is a parallel group under MPI.
        """
        # connect regular_phases with each other if you are optimizing alt or mach
        prob._link_phases_helper_with_options(
            prob.regular_phases,
            'optimize_altitude',
            Dynamic.Mission.ALTITUDE,
            ref=1.0e4,
        )
        prob._link_phases_helper_with_options(
            prob.regular_phases, 'optimize_mach', Dynamic.Atmosphere.MACH
        )

        # connect reserve phases with each other if you are optimizing alt or mach
        prob._link_phases_helper_with_options(
            prob.reserve_phases,
            'optimize_altitude',
            Dynamic.Mission.ALTITUDE,
            ref=1.0e4,
        )
        prob._link_phases_helper_with_options(
            prob.reserve_phases, 'optimize_mach', Dynamic.Atmosphere.MACH
        )

        # connect mass and distance between all phases regardless of reserve /
        # non-reserve status
        prob.traj.link_phases(
            phases, ['time'], ref=None if connect_directly else 1e3, connected=connect_directly
        )
        prob.traj.link_phases(
            phases,
            [Dynamic.Vehicle.MASS],
            ref=None if connect_directly else 1e6,
            connected=connect_directly,
        )
        prob.traj.link_phases(
            phases,
            [Dynamic.Mission.DISTANCE],
            ref=None if connect_directly else 1e3,
            connected=connect_directly,
        )

        prob.model.connect(
            f'traj.{prob.regular_phases[-1]}.timeseries.distance',
            Mission.Summary.RANGE,
            src_indices=[-1],
            flat_src_indices=True,
        )

    def add_post_mission_systems(self, prob, include_landing=True):
        """
        Add any post mission systems.

        These may include any post-mission take off and landing systems.

        This is called from AviaryProblem.add_post_mission_systems

        Parameters
        ----------
        prob : AviaryProblem
            Problem that owns this builder.
        include_landing : bool
            When True, include the landing systems.
        """
        if prob.pre_mission_info['include_takeoff']:
            self._add_post_mission_takeoff_systems(prob)
        else:
            first_flight_phase_name = list(prob.phase_info.keys())[0]
            first_flight_phase = prob.traj._phases[first_flight_phase_name]
            first_flight_phase.set_state_options(Dynamic.Vehicle.MASS, fix_initial=False)

        if include_landing and prob.post_mission_info['include_landing']:
            self._add_landing_systems(prob)

        # connect summary mass to the initial guess of mass in the first phase
        if not prob.pre_mission_info['include_takeoff']:
            first_flight_phase_name = list(prob.phase_info.keys())[0]

            eq = prob.model.add_subsystem(
                f'link_{first_flight_phase_name}_mass',
                om.EQConstraintComp(),
                promotes_inputs=[('rhs:mass', Mission.Summary.GROSS_MASS)],
            )

            eq.add_eq_output(
                'mass', eq_units='lbm', normalize=False, ref=100000.0, add_constraint=True
            )

            prob.model.connect(
                f'traj.{first_flight_phase_name}.states:mass',
                f'link_{first_flight_phase_name}_mass.lhs:mass',
                src_indices=[0],
                flat_src_indices=True,
            )

        prob.model.add_subsystem(
            'range_constraint',
            om.ExecComp(
                'range_resid = target_range - actual_range',
                target_range={'val': prob.target_range, 'units': 'NM'},
                actual_range={'val': prob.target_range, 'units': 'NM'},
                range_resid={'val': 30, 'units': 'NM'},
            ),
            promotes_inputs=[
                ('actual_range', Mission.Summary.RANGE),
                'target_range',
            ],
            promotes_outputs=[('range_resid', Mission.Constraints.RANGE_RESIDUAL)],
        )

        prob.post_mission.add_constraint(Mission.Constraints.MASS_RESIDUAL, equals=0.0, ref=1.0e5)

    def _add_post_mission_takeoff_systems(self, prob):
        first_flight_phase_name = list(prob.phase_info.keys())[0]
        connect_takeoff_to_climb = not prob.phase_info[first_flight_phase_name]['user_options'].get(
            'add_initial_mass_constraint', True
        )

        if connect_takeoff_to_climb:
            prob.model.connect(
                Mission.Takeoff.FINAL_MASS, f'traj.{first_flight_phase_name}.initial_states:mass'
            )
            prob.model.connect(
                Mission.Takeoff.GROUND_DISTANCE,
                f'traj.{first_flight_phase_name}.initial_states:distance',
            )

            control_type_string = 'control_values'
            if prob.phase_info[first_flight_phase_name]['user_options'].get(
                'use_polynomial_control', True
            ):
                if not use_new_dymos_syntax:
                    control_type_string = 'polynomial_control_values'

            if prob.phase_info[first_flight_phase_name]['user_options'].get('optimize_mach', False):
                # Create an ExecComp to compute the difference in mach
                mach_diff_comp = om.ExecComp(
                    'mach_resid_for_connecting_takeoff = final_mach - initial_mach'
                )
                prob.model.add_subsystem('mach_diff_comp', mach_diff_comp)

                # Connect the inputs to the mach difference component
                prob.model.connect(Mission.Takeoff.FINAL_MACH, 'mach_diff_comp.final_mach')
                prob.model.connect(
                    f'traj.{first_flight_phase_name}.{control_type_string}:mach',
                    'mach_diff_comp.initial_mach',
                    src_indices=[0],
                )

                # Add constraint for mach difference
                prob.model.add_constraint(
                    'mach_diff_comp.mach_resid_for_connecting_takeoff', equals=0.0
                )

            if prob.phase_info[first_flight_phase_name]['user_options'].get(
                'optimize_altitude', False
            ):
                # Similar steps for altitude difference
                alt_diff_comp = om.ExecComp(
                    'altitude_resid_for_connecting_takeoff = final_altitude - initial_altitude',
                    units='ft',
                )
                prob.model.add_subsystem('alt_diff_comp', alt_diff_comp)

                prob.model.connect(Mission.Takeoff.FINAL_ALTITUDE, 'alt_diff_comp.final_altitude')
                prob.model.connect(
                    f'traj.{first_flight_phase_name}.{control_type_string}:altitude',
                    'alt_diff_comp.initial_altitude',
                    src_indices=[0],
                )

                prob.model.add_constraint(
                    'alt_diff_comp.altitude_resid_for_connecting_takeoff', equals=0.0
                )

    def _add_landing_systems(self, prob):
        landing_options = Landing(
            ref_wing_area=prob.aviary_inputs.get_val(Aircraft.Wing.AREA, units='ft**2'),
            Cl_max_ldg=prob.aviary_inputs.get_val(Mission.Landing.LIFT_COEFFICIENT_MAX),  # no units
        )

        landing = landing_options.build_phase(False)

        prob.model.add_subsystem(
            'landing',
            landing,
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=['mission:*'],
        )

        last_flight_phase_name = list(prob.phase_info.keys())[-1]

        control_type_string = 'control_values'
        if prob.phase_info[last_flight_phase_name]['user_options'].get(
            'use_polynomial_control', True
        ):
            if not use_new_dymos_syntax:
                control_type_string = 'polynomial_control_values'

        last_regular_phase = prob.regular_phases[-1]
        prob.model.connect(
            f'traj.{last_regular_phase}.states:mass',
            Mission.Landing.TOUCHDOWN_MASS,
            src_indices=[-1],
        )
        prob.model.connect(
            f'traj.{last_regular_phase}.{control_type_string}:altitude',
            Mission.Landing.INITIAL_ALTITUDE,
            src_indices=[0],
        )

    def add_objective(self, prob):
        """
        Add any additional components related to objectives.

        Parameters
        ----------
        prob : AviaryProblem
            Problem that owns this builder.
        """
        pass

    def add_guesses(self, prob, phase_name, phase, guesses, target_prob, parent_prefix):
        """
        Adds the initial guesses for each variable of a given phase to the problem.
        This method sets the initial guesses for time, control, state, and problem-specific
        variables for a given phase. If using the GASP model, it also handles some special
        cases that are not covered in the `phase_info` object. These include initial guesses
        for mass, time, and distance, which are determined based on the phase name and other
        mission-related variables.

        Parameters
        ----------
        phase_name : str
            The name of the phase for which the guesses are being added.
        phase : Phase
            The phase object for which the guesses are being added.
        guesses : dict
            A dictionary containing the initial guesses for the phase.
        target_prob : Problem
            Problem instance to apply the guesses.
        parent_prefix : str
            Location of this trajectory in the hierarchy.
        """
        control_keys = ['mach', 'altitude']
        state_keys = ['mass', Dynamic.Mission.DISTANCE]

        prob_keys = ['tau_gear', 'tau_flaps']

        # for the simple mission method, use the provided initial and final mach
        # and altitude values from phase_info
        initial_altitude = wrapped_convert_units(
            prob.phase_info[phase_name]['user_options']['initial_altitude'], 'ft'
        )
        final_altitude = wrapped_convert_units(
            prob.phase_info[phase_name]['user_options']['final_altitude'], 'ft'
        )
        initial_mach = prob.phase_info[phase_name]['user_options']['initial_mach']
        final_mach = prob.phase_info[phase_name]['user_options']['final_mach']

        guesses['mach'] = ([initial_mach[0], final_mach[0]], 'unitless')
        guesses['altitude'] = ([initial_altitude, final_altitude], 'ft')

        # if time not in initial guesses, set it to the average of the
        # initial_bounds and the duration_bounds
        if 'time' not in guesses:
            initial_bounds = wrapped_convert_units(
                prob.phase_info[phase_name]['user_options']['initial_bounds'], 's'
            )
            duration_bounds = wrapped_convert_units(
                prob.phase_info[phase_name]['user_options']['duration_bounds'], 's'
            )
            guesses['time'] = ([np.mean(initial_bounds[0]), np.mean(duration_bounds[0])], 's')

        # if time not in initial guesses, set it to the average of the
        # initial_bounds and the duration_bounds
        if 'time' not in guesses:
            initial_bounds = prob.phase_info[phase_name]['user_options']['initial_bounds']
            duration_bounds = prob.phase_info[phase_name]['user_options']['duration_bounds']
            # Add a check for the initial and duration bounds, raise an error if they
            # are not consistent
            if initial_bounds[1] != duration_bounds[1]:
                raise ValueError(
                    f'Initial and duration bounds for {phase_name} are not consistent.'
                )
            guesses['time'] = (
                [np.mean(initial_bounds[0]), np.mean(duration_bounds[0])],
                initial_bounds[1],
            )

        for guess_key, guess_data in guesses.items():
            val, units = guess_data

            # Set initial guess for time variables
            if 'time' == guess_key:
                target_prob.set_val(
                    parent_prefix + f'traj.{phase_name}.t_initial', val[0], units=units
                )
                target_prob.set_val(
                    parent_prefix + f'traj.{phase_name}.t_duration', val[1], units=units
                )

            else:
                # Set initial guess for control variables
                if guess_key in control_keys:
                    try:
                        target_prob.set_val(
                            parent_prefix + f'traj.{phase_name}.controls:{guess_key}',
                            prob._process_guess_var(val, guess_key, phase),
                            units=units,
                        )

                    except KeyError:
                        try:
                            target_prob.set_val(
                                parent_prefix
                                + f'traj.{phase_name}.polynomial_controls:{guess_key}',
                                prob._process_guess_var(val, guess_key, phase),
                                units=units,
                            )

                        except KeyError:
                            target_prob.set_val(
                                parent_prefix + f'traj.{phase_name}.bspline_controls:',
                                {guess_key},
                                prob._process_guess_var(val, guess_key, phase),
                                units=units,
                            )

                if guess_key in control_keys:
                    pass

                # Set initial guess for state variables
                elif guess_key in state_keys:
                    target_prob.set_val(
                        parent_prefix + f'traj.{phase_name}.states:{guess_key}',
                        prob._process_guess_var(val, guess_key, phase),
                        units=units,
                    )

                elif guess_key in prob_keys:
                    target_prob.set_val(parent_prefix + guess_key, val, units=units)

                elif ':' in guess_key:
                    target_prob.set_val(
                        parent_prefix + f'traj.{phase_name}.{guess_key}',
                        prob._process_guess_var(val, guess_key, phase),
                        units=units,
                    )
                else:
                    # raise error if the guess key is not recognized
                    raise ValueError(
                        f'Initial guess key {guess_key} in {phase_name} is not recognized.'
                    )

        if 'mass' not in guesses:
            mass_guess = prob.aviary_inputs.get_val(Mission.Design.GROSS_MASS, units='lbm')

            # Set the mass guess as the initial value for the mass state variable
            target_prob.set_val(
                parent_prefix + f'traj.{phase_name}.states:mass', mass_guess, units='lbm'
            )
