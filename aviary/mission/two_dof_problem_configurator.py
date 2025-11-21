import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM, RHO_SEA_LEVEL_ENGLISH
from aviary.mission.gasp_based.ode.landing_ode import LandingSegment
from aviary.mission.gasp_based.ode.params import ParamPort
from aviary.mission.gasp_based.ode.taxi_ode import TaxiSegment
from aviary.mission.gasp_based.phases.accel_phase import AccelPhase
from aviary.mission.gasp_based.phases.ascent_phase import AscentPhase
from aviary.mission.gasp_based.phases.climb_phase import ClimbPhase
from aviary.mission.gasp_based.phases.cruise_phase import CruisePhase
from aviary.mission.gasp_based.phases.descent_phase import DescentPhase
from aviary.mission.gasp_based.phases.groundroll_phase import GroundrollPhase
from aviary.mission.gasp_based.phases.rotation_phase import RotationPhase
from aviary.mission.gasp_based.polynomial_fit import PolynomialFit
from aviary.mission.problem_configurator import ProblemConfiguratorBase
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.functions import add_opts2vals, create_opts2vals
from aviary.utils.process_input_decks import initialization_guessing, update_GASP_options
from aviary.utils.utils import wrapped_convert_units
from aviary.variable_info.enums import LegacyCode
from aviary.variable_info.variables import Aircraft, Dynamic, Mission
from aviary.mission.utils import process_guess_var


class TwoDOFProblemConfigurator(ProblemConfiguratorBase):
    """
    A 2DOF specific builder that customizes AviaryProblem() for use with
     two degree of freedom phases.
    """

    def initial_guesses(self, aviary_group):
        """
        Set any initial guesses for variables in the aviary problem.

        This is called at the end of AivaryProblem.load_inputs.

        Parameters
        ----------
        aviary_group : AviaryGroup
            Aviary model that owns this configurator.
        """
        # TODO: This should probably be moved to the set_initial_guesses() method in AviaryProblem class
        # Defines how the problem should build it's initial guesses for load_inputs()
        # this modifies mass_method, initialization_guesses, and aviary_values

        aviary_inputs = aviary_group.aviary_inputs

        aviary_inputs = update_GASP_options(aviary_inputs)

        if aviary_group.engine_builders is None:
            aviary_group.engine_builders = [build_engine_deck(aviary_inputs)]

        aviary_group.initialization_guesses = initialization_guessing(
            aviary_inputs, aviary_group.initialization_guesses, aviary_group.engine_builders
        )

        aviary_inputs.set_val(
            Mission.Summary.CRUISE_MASS_FINAL,
            val=aviary_group.initialization_guesses['cruise_mass_final'],
            units='lbm',
        )
        aviary_inputs.set_val(
            Mission.Summary.GROSS_MASS,
            val=aviary_group.initialization_guesses['actual_takeoff_mass'],
            units='lbm',
        )

        # Deal with missing defaults in phase info:
        aviary_group.pre_mission_info.setdefault('include_takeoff', True)
        aviary_group.pre_mission_info.setdefault('external_subsystems', [])

        aviary_group.post_mission_info.setdefault('include_landing', True)
        aviary_group.post_mission_info.setdefault('external_subsystems', [])

        # Commonly referenced values
        aviary_group.cruise_alt = aviary_inputs.get_val(Mission.Design.CRUISE_ALTITUDE, units='ft')
        aviary_group.mass_defect = aviary_inputs.get_val('mass_defect', units='lbm')

        aviary_group.cruise_mass_final = aviary_inputs.get_val(
            Mission.Summary.CRUISE_MASS_FINAL, units='lbm'
        )

        if 'target_range' in aviary_group.post_mission_info:
            aviary_group.target_range = wrapped_convert_units(
                aviary_group.post_mission_info['target_range'], 'NM'
            )
            aviary_inputs.set_val(Mission.Summary.RANGE, aviary_group.target_range, units='NM')
        else:
            aviary_group.target_range = aviary_inputs.get_val(Mission.Design.RANGE, units='NM')
            aviary_inputs.set_val(
                Mission.Summary.RANGE,
                aviary_inputs.get_val(Mission.Design.RANGE, units='NM'),
                units='NM',
            )

        aviary_group.cruise_mach = aviary_inputs.get_val(Mission.Design.MACH)
        aviary_group.require_range_residual = True

    def get_default_phase_info(self, aviary_group):
        """
        Return a default phase_info for this type or problem.

        The default phase_info is used in the level 1 and 2 interfaces when no
        phase_info is specified.

        This is called during load_inputs.

        Parameters
        ----------
        aviary_group : AviaryGroup
            Aviary model that owns this configurator.

        Returns
        -------
        AviaryValues
            General default phase_info.
        """
        from aviary.models.missions.two_dof_default import phase_info

        return phase_info

    def get_code_origin(self, aviary_group):
        """
        Return the legacy of this problem configurator.

        Parameters
        ----------
        aviary_group : AviaryGroup
            Aviary model that owns this configurator.

        Returns
        -------
        LegacyCode
            Code origin enum.
        """
        return LegacyCode.GASP

    def add_takeoff_systems(self, aviary_group):
        """
        Adds takeoff systems to the model in aviary_group.

        Parameters
        ----------
        aviary_group : AviaryProblem
            Problem that owns this configurator.
        """
        aviary_group.cruise_alt = aviary_group.aviary_inputs.get_val(
            Mission.Design.CRUISE_ALTITUDE, units='ft'
        )

        # Add event transformation subsystem
        aviary_group.add_subsystem(
            'event_xform',
            om.ExecComp(
                ['t_init_gear=m*tau_gear+b', 't_init_flaps=m*tau_flaps+b'],
                t_init_gear={'units': 's'},  # initial time that gear comes up
                t_init_flaps={'units': 's'},  # initial time that flaps retract
                tau_gear={'units': 'unitless'},
                tau_flaps={'units': 'unitless'},
                m={'units': 's'},
                b={'units': 's'},
            ),
            promotes_inputs=[
                'tau_gear',  # design var
                'tau_flaps',  # design var
                ('m', Mission.Takeoff.ASCENT_DURATION),
                ('b', Mission.Takeoff.ASCENT_T_INITIAL),
            ],
            promotes_outputs=['t_init_gear', 't_init_flaps'],  # link to h_fit
        )

        # Add taxi subsystem
        aviary_group.add_subsystem(
            'taxi',
            TaxiSegment(**(aviary_group.ode_args)),
            promotes_inputs=['aircraft:*', 'mission:*'],
        )

        # Calculate speed at which to initiate rotation
        aviary_group.add_subsystem(
            'vrot',
            om.ExecComp(
                'Vrot = ((2 * mass * g) / (rho * wing_area * CLmax))**0.5 + dV1 + dVR',
                Vrot={'units': 'ft/s'},
                mass={'units': 'lbm'},
                CLmax={'units': 'unitless'},
                g={'units': 'lbf/lbm', 'val': GRAV_ENGLISH_LBM},
                rho={'units': 'slug/ft**3', 'val': RHO_SEA_LEVEL_ENGLISH},
                wing_area={'units': 'ft**2'},
                dV1={
                    'units': 'ft/s',
                    'desc': 'Increment of engine failure decision speed above stall',
                },
                dVR={
                    'units': 'ft/s',
                    'desc': 'Increment of takeoff rotation speed above engine failure '
                    'decision speed',
                },
            ),
            promotes_inputs=[
                ('wing_area', Aircraft.Wing.AREA),
                ('dV1', Mission.Takeoff.DECISION_SPEED_INCREMENT),
                ('dVR', Mission.Takeoff.ROTATION_SPEED_INCREMENT),
                ('CLmax', Mission.Takeoff.LIFT_COEFFICIENT_MAX),
            ],
            promotes_outputs=[('Vrot', Mission.Takeoff.ROTATION_VELOCITY)],
        )

    def get_phase_builder(self, aviary_group, phase_name, phase_options):
        """
        Return a phase_builder for the requested phase.

        This is called from _get_phase in AviaryProblem.add_phases

        Parameters
        ----------
        aviary_group : AviaryGroup
            Aviary model that owns this configurator.
        phase_name : str
            Name of the requested phase.
        phase_options : dict
            Phase options for the requested phase.

        Returns
        -------
        PhaseBuilderBase
            Phase builder for requested phase.
        """
        if 'groundroll' in phase_name:
            phase_builder = GroundrollPhase
        elif 'rotation' in phase_name:
            phase_builder = RotationPhase
        elif 'accel' in phase_name:
            phase_builder = AccelPhase
        elif 'ascent' in phase_name:
            phase_builder = AscentPhase
        elif 'climb' in phase_name:
            phase_builder = ClimbPhase
        elif 'cruise' in phase_name:
            phase_builder = CruisePhase
        elif 'desc' in phase_name:
            phase_builder = DescentPhase
        else:
            raise ValueError(
                f'{phase_name} does not have an associated phase_builder \n phase_name must '
                'include one of: groundroll, rotation, accel, ascent, climb, cruise, or desc'
            )

        return phase_builder

    def set_phase_options(self, aviary_group, phase_name, phase_idx, phase, user_options, comm):
        """
        Set any necessary problem-related options on the phase.

        This is called from _get_phase in AviaryProblem.add_phases

        Parameters
        ----------
        aviary_group : AviaryGroup
            Aviary model that owns this configurator.
        phase_name : str
            Name of the requested phase.
        phase_idx : int
            Phase position in aviary_group.phases. Can be used to identify first phase.
        phase : Phase
            Instantiated phase object.
        user_options : dict
            Subdictionary "user_options" from the phase_info.
        comm : MPI.Comm or <FakeComm>
            MPI Communicator from OpenMDAO problem.
        """
        if 'cruise' in phase_name:
            # Cruise phase integrates over mass instead of time.
            # We rely on mass being monotonically non-increasing across the phase.
            phase.set_time_options(
                name='mass',
                fix_initial=False,
                fix_duration=False,
                units='lbm',
                targets='mass',
                initial_bounds=(0.0, 1.0e7),
                initial_ref=100.0e3,
                duration_bounds=(-1.0e7, -1),
                duration_ref=50000,
            )
            return

        time_units = 's'
        initial = wrapped_convert_units(user_options['time_initial'], time_units)
        duration = wrapped_convert_units(user_options['time_duration'], time_units)
        initial_bounds = wrapped_convert_units(user_options['time_initial_bounds'], time_units)
        duration_bounds = wrapped_convert_units(user_options['time_duration_bounds'], time_units)
        initial_ref = wrapped_convert_units(user_options['time_initial_ref'], time_units)
        duration_ref = wrapped_convert_units(user_options['time_duration_ref'], time_units)

        fix_duration = duration is not None
        fix_initial = initial is not None

        if 'ascent' in phase_name:
            phase.set_time_options(
                units='s',
                targets='t_curr',
                input_initial=True,
                input_duration=True,
            )

        elif 'descent' in phase_name:
            phase.set_time_options(
                duration_bounds=duration_bounds,
                fix_initial=fix_initial,
                input_initial=input_initial,
                units='s',
                duration_ref=duration_ref,
            )

        else:
            if initial_bounds[0] is not None and initial_bounds[1]:
                # Upper bound is good for a ref.
                time_initial_ref = initial_bounds[1]
            else:
                time_initial_ref = 600.0

            if duration_ref is None:
                # Why are we overriding this?
                duration_ref = 0.5 * (duration_bounds[0] + duration_bounds[1])

            input_initial = phase_idx > 0

            if fix_initial or input_initial:
                if comm.size > 1:
                    # Phases are disconnected to run in parallel, so initial ref is
                    # valid.
                    initial_ref = time_initial_ref
                else:
                    # Redundant on a fixed input; raises a warning if specified.
                    initial_ref = None

                phase.set_time_options(
                    fix_initial=fix_initial,
                    fix_duration=fix_duration,
                    units=time_units,
                    duration_bounds=duration_bounds,
                    duration_ref=duration_ref,
                    initial_ref=initial_ref,
                )

            else:  # TODO: figure out how to handle this now that fix_initial is dict
                phase.set_time_options(
                    fix_initial=fix_initial,
                    fix_duration=fix_duration,
                    units=time_units,
                    duration_bounds=duration_bounds,
                    duration_ref=duration_ref,
                    initial_bounds=initial_bounds,
                    initial_ref=initial_ref,
                )

        phase.add_control(
            Dynamic.Vehicle.Propulsion.THROTTLE,
            targets=Dynamic.Vehicle.Propulsion.THROTTLE,
            units='unitless',
            opt=False,
        )

        # TODO: This seems like a hack. We might want to find a better way.
        #       The issue is that aero methods are hardcoded for GASP mission phases
        #       instead of being defaulted somewhere, so they don't use phase_info
        # aviary_group.mission_info[phase_name]['phase_type'] = phase_name
        if phase_name in ['ascent', 'groundroll', 'rotation']:
            # safely add in default method in way that doesn't overwrite existing method
            # and create nested structure if it doesn't already exist
            aviary_group.mission_info[phase_name].setdefault('subsystem_options', {}).setdefault(
                'core_aerodynamics', {}
            ).setdefault('method', 'low_speed')

    def link_phases(self, aviary_group, phases, connect_directly=True):
        """
        Apply any additional phase linking.

        Note that some phase variables are handled in the AviaryProblem. Only
        problem-specific ones need to be linked here.

        This is called from AviaryProblem.link_phases

        Parameters
        ----------
        aviary_group : AviaryGroup
            Aviary model that owns this configurator.
        phases : Phase
            Phases to be linked.
        connect_directly : bool
            When True, then connected=True. This allows the connections to be
            handled by constraints if `phases` is a parallel group under MPI.
        """
        for ii in range(len(phases) - 1):
            phase1, phase2 = phases[ii : ii + 2]
            analytic1 = aviary_group.mission_info[phase1]['user_options']['analytic']
            analytic2 = aviary_group.mission_info[phase2]['user_options']['analytic']

            if not (analytic1 or analytic2):
                # we always want time, distance, and mass to be continuous
                states_to_link = {
                    'time': connect_directly,
                    Dynamic.Mission.DISTANCE: connect_directly,
                    Dynamic.Vehicle.MASS: False,
                }

                # if both phases are reserve phases or neither is a reserve phase
                # (we are not on the boundary between the regular and reserve missions)
                # and neither phase is ground roll or rotation (altitude isn't a state):
                # we want altitude to be continuous as well
                if (
                    (
                        (phase1 in aviary_group.reserve_phases)
                        == (phase2 in aviary_group.reserve_phases)
                    )
                    and not ({'groundroll', 'rotation'} & {phase1, phase2})
                    and not ('accel', 'climb1') == (phase1, phase2)
                ):  # required for convergence of FwGm
                    states_to_link[Dynamic.Mission.ALTITUDE] = connect_directly

                # if either phase is rotation, we need to connect velocity
                # ascent to accel also requires velocity
                if 'rotation' in (phase1, phase2) or ('ascent', 'accel') == (phase1, phase2):
                    states_to_link[Dynamic.Mission.VELOCITY] = connect_directly
                    # if the first phase is rotation, we also need alpha
                    if phase1 == 'rotation':
                        states_to_link[Dynamic.Vehicle.ANGLE_OF_ATTACK] = False

                for state, connected in states_to_link.items():
                    # in initial guesses, all of the states, other than time use
                    # the same name
                    initial_guesses1 = aviary_group.mission_info[phase1]['initial_guesses']
                    initial_guesses2 = aviary_group.mission_info[phase2]['initial_guesses']

                    # if a state is in the initial guesses, get the units of the
                    # initial guess
                    kwargs = {}
                    if not connected:
                        if state in initial_guesses1:
                            kwargs = {'units': initial_guesses1[state][-1]}
                        elif state in initial_guesses2:
                            kwargs = {'units': initial_guesses2[state][-1]}

                    aviary_group.traj.link_phases(
                        [phase1, phase2], [state], connected=connected, **kwargs
                    )

            # if either phase is analytic we have to use a linkage_constraint
            else:
                # analytic phases use the prefix "initial" for time and distance,
                # but not mass
                if analytic2:
                    prefix = 'initial_'
                else:
                    prefix = ''

                aviary_group.traj.add_linkage_constraint(
                    phase1, phase2, 'time', prefix + 'time', connected=True
                )
                aviary_group.traj.add_linkage_constraint(
                    phase1, phase2, 'distance', prefix + 'distance', connected=True
                )
                aviary_group.traj.add_linkage_constraint(
                    phase1, phase2, 'mass', 'mass', connected=False, ref=1.0e5
                )

                # This isn't computed, but is instead set in the cruise phase_info.
                # We still need altitude continuity.
                # Note: if both sides are Breguet Range, the user is doing something odd like a
                # step cruise, so don't enforce a constraint.
                if not (analytic1 and analytic2):
                    aviary_group.traj.add_linkage_constraint(
                        phase1, phase2, 'altitude', 'altitude', connected=False, ref=1.0e4
                    )

        # add all params and promote them to aviary_group level
        ParamPort.promote_params(
            aviary_group,
            trajs=['traj'],
            phases=[[*aviary_group.regular_phases, *aviary_group.reserve_phases]],
        )

        aviary_group.promotes(
            'traj',
            inputs=[
                ('ascent.parameters:t_init_gear', 't_init_gear'),
                ('ascent.parameters:t_init_flaps', 't_init_flaps'),
                ('ascent.t_initial', Mission.Takeoff.ASCENT_T_INITIAL),
                ('ascent.t_duration', Mission.Takeoff.ASCENT_DURATION),
            ],
        )

        # imitate input_initial for taxi -> groundroll
        eq = aviary_group.add_subsystem('taxi_groundroll_mass_constraint', om.EQConstraintComp())
        eq.add_eq_output('mass', eq_units='lbm', normalize=False, ref=10000.0, add_constraint=True)
        aviary_group.connect('taxi.mass', 'taxi_groundroll_mass_constraint.rhs:mass')
        aviary_group.connect(
            'traj.groundroll.states:mass',
            'taxi_groundroll_mass_constraint.lhs:mass',
            src_indices=[0],
            flat_src_indices=True,
        )

        aviary_group.connect('traj.ascent.timeseries.time', 'h_fit.time_cp')
        aviary_group.connect('traj.ascent.timeseries.altitude', 'h_fit.h_cp')

        aviary_group.connect(
            f'traj.{aviary_group.regular_phases[-1]}.states:mass',
            Mission.Landing.TOUCHDOWN_MASS,
            src_indices=[-1],
        )

        # promote all ParamPort inputs for analytic segments as well
        param_list = list(ParamPort.param_data)
        aviary_group.promotes('taxi', inputs=param_list)
        aviary_group.promotes('landing', inputs=param_list)
        aviary_group.connect('taxi.mass', 'vrot.mass')

        if 'ascent' in aviary_group.mission_info:
            self._add_groundroll_eq_constraint(aviary_group)

    def check_trajectory(self, aviary_group):
        """
        Checks the phase_info user options for any inconsistency.

        Parameters
        ----------
        aviary_group : AviaryGroup
            Aviary model that owns this configurator.
        """
        pass

    def _add_groundroll_eq_constraint(self, aviary_group):
        """
        Add an equality constraint to the problem to ensure that the TAS at the end of the
        groundroll phase is equal to the rotation velocity at the start of the rotation phase.
        """
        aviary_group.add_subsystem(
            'groundroll_boundary',
            om.EQConstraintComp(
                'velocity',
                eq_units='ft/s',
                normalize=True,
                add_constraint=True,
            ),
        )
        aviary_group.connect(Mission.Takeoff.ROTATION_VELOCITY, 'groundroll_boundary.rhs:velocity')
        aviary_group.connect(
            'traj.groundroll.states:velocity',
            'groundroll_boundary.lhs:velocity',
            src_indices=[-1],
            flat_src_indices=True,
        )

    def add_post_mission_systems(self, aviary_group):
        """
        Add any post mission systems.

        These may include any post-mission take off and landing systems.

        This is called from AviaryProblem.add_post_mission_systems

        Parameters
        ----------
        aviary_group : AviaryGroup
            Aviary model that owns this configurator.
        """
        if aviary_group.post_mission_info['include_landing']:
            self._add_landing_systems(aviary_group)

        ascent_phase = getattr(aviary_group.traj.phases, 'ascent')
        ascent_tx = ascent_phase.options['transcription']
        ascent_num_nodes = ascent_tx.grid_data.num_nodes
        aviary_group.add_subsystem(
            'h_fit',
            PolynomialFit(N_cp=ascent_num_nodes),
            promotes_inputs=['t_init_gear', 't_init_flaps'],
        )

        aviary_group.add_subsystem(
            'range_constraint',
            om.ExecComp(
                'range_resid = target_range - actual_range',
                target_range={'val': aviary_group.target_range, 'units': 'NM'},
                actual_range={'val': aviary_group.target_range, 'units': 'NM'},
                range_resid={'val': 30, 'units': 'NM'},
            ),
            promotes_inputs=[
                ('actual_range', Mission.Summary.RANGE),
                'target_range',
            ],
            promotes_outputs=[('range_resid', Mission.Constraints.RANGE_RESIDUAL)],
        )

        aviary_group.post_mission.add_constraint(
            Mission.Constraints.MASS_RESIDUAL, equals=0.0, ref=1.0e5
        )

    def _add_landing_systems(self, aviary_group):
        aviary_group.add_subsystem(
            'landing',
            LandingSegment(**(aviary_group.ode_args)),
            promotes_inputs=[
                'aircraft:*',
                'mission:*',
                (Dynamic.Vehicle.MASS, Mission.Landing.TOUCHDOWN_MASS),
            ],
            promotes_outputs=['mission:*'],
        )

        aviary_group.connect(
            'pre_mission.interference_independent_of_shielded_area',
            'landing.interference_independent_of_shielded_area',
        )
        aviary_group.connect(
            'pre_mission.drag_loss_due_to_shielded_wing_area',
            'landing.drag_loss_due_to_shielded_wing_area',
        )

    def set_phase_initial_guesses(
        self, aviary_group, phase_name, phase, guesses, target_prob, parent_prefix
    ):
        """
        Adds the initial guesses for each variable of a given phase to the problem.

        This method sets the initial guesses into the openmdao model for time, controls, states,
        and problem-specific variables for a given phase. If using the GASP model, it also handles
        some special cases that are not covered in the `phase_info` object. These include initial
        guesses for mass, time, and distance, which are determined based on the phase name and
        other mission-related variables.

        Parameters
        ----------
        aviary_group : AviaryGroup
            Aviary model that owns this configurator.
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
        # Handle Analytic Phase
        if aviary_group.mission_info[phase_name]['user_options'].get('analytic', False):
            for guess_key, guess_data in guesses.items():
                val, units = guess_data

                if 'mass' == guess_key:
                    # Set initial and duration mass for the analytic cruise phase.
                    # Note we are integrating over mass, not time for this phase.
                    target_prob.set_val(
                        parent_prefix + f'traj.{phase_name}.t_initial', val[0], units=units
                    )
                    target_prob.set_val(
                        parent_prefix + f'traj.{phase_name}.t_duration', val[1], units=units
                    )

                elif guess_key == 'time':
                    # Time needs to be dealt with elsewhere.
                    continue
                else:
                    # Otherwise, set the value of the parameter in the trajectory
                    # phase
                    target_prob.set_val(
                        parent_prefix + f'traj.{phase_name}.parameters:{guess_key}',
                        val,
                        units=units,
                    )

            # Analytic phase should have nothing else to set.
            return

        # Set initial guesses for the rotation mass and flight duration
        rotation_mass = aviary_group.initialization_guesses['rotation_mass']
        flight_duration = aviary_group.initialization_guesses['flight_duration']

        control_keys = ['velocity_rate', 'throttle']
        state_keys = [
            'altitude',
            'mass',
            Dynamic.Mission.DISTANCE,
            Dynamic.Mission.VELOCITY,
            'flight_path_angle',
            Dynamic.Vehicle.ANGLE_OF_ATTACK,
        ]

        if phase_name == 'ascent':
            # Alpha is a control for ascent.
            control_keys.append(Dynamic.Vehicle.ANGLE_OF_ATTACK)

        prob_keys = ['tau_gear', 'tau_flaps']

        for guess_key, guess_data in guesses.items():
            val, units = guess_data

            # Set initial guess for time variables
            if 'time' == guess_key:
                if phase_name == 'ascent':
                    # These variables are promoted to the top.
                    init_path = Mission.Takeoff.ASCENT_T_INITIAL
                    dura_path = Mission.Takeoff.ASCENT_DURATION
                else:
                    init_path = parent_prefix + f'traj.{phase_name}.t_initial'
                    dura_path = parent_prefix + f'traj.{phase_name}.t_duration'

                target_prob.set_val(init_path, val[0], units=units)
                target_prob.set_val(dura_path, val[1], units=units)

            else:
                # Set initial guess for control variables
                if guess_key in control_keys:
                    try:
                        target_prob.set_val(
                            parent_prefix + f'traj.{phase_name}.controls:{guess_key}',
                            process_guess_var(val, guess_key, phase),
                            units=units,
                        )

                    except KeyError:
                        try:
                            target_prob.set_val(
                                parent_prefix
                                + f'traj.{phase_name}.polynomial_controls:{guess_key}',
                                process_guess_var(val, guess_key, phase),
                                units=units,
                            )

                        except KeyError:
                            target_prob.set_val(
                                parent_prefix + f'traj.{phase_name}.bspline_controls:',
                                {guess_key},
                                process_guess_var(val, guess_key, phase),
                                units=units,
                            )

                if guess_key in control_keys:
                    pass

                # Set initial guess for state variables
                elif guess_key in state_keys:
                    target_prob.set_val(
                        parent_prefix + f'traj.{phase_name}.states:{guess_key}',
                        process_guess_var(val, guess_key, phase),
                        units=units,
                    )

                elif guess_key in prob_keys:
                    target_prob.set_val(parent_prefix + guess_key, val, units=units)

                elif ':' in guess_key:
                    target_prob.set_val(
                        parent_prefix + f'traj.{phase_name}.{guess_key}',
                        process_guess_var(val, guess_key, phase),
                        units=units,
                    )
                else:
                    # raise error if the guess key is not recognized
                    raise ValueError(
                        f'Initial guess key {guess_key} in {phase_name} is not recognized.'
                    )

        # We need some special logic for these following variables because GASP computes
        # initial guesses using some knowledge of the mission duration and other variables
        # that are only available after calling `create_vehicle`. Thus these initial guess
        # values are not included in the `phase_info` object.
        base_phase = phase_name.removeprefix('reserve_')

        if 'mass' not in guesses:
            # Determine a mass guess depending on the phase name
            if base_phase in ['groundroll', 'rotation', 'ascent', 'accel', 'climb1']:
                mass_guess = rotation_mass
            elif base_phase == 'climb2':
                mass_guess = 0.99 * rotation_mass
            elif 'desc' in base_phase:
                mass_guess = 0.9 * aviary_group.cruise_mass_final

            # Set the mass guess as the initial value for the mass state variable
            target_prob.set_val(
                parent_prefix + f'traj.{phase_name}.states:mass', mass_guess, units='lbm'
            )

        if 'time' not in guesses:
            # Determine initial time and duration guesses depending on the phase name
            if 'desc1' == base_phase:
                t_initial = flight_duration * 0.9
                t_duration = flight_duration * 0.04
            elif 'desc2' in base_phase:
                t_initial = flight_duration * 0.94
                t_duration = 5000

            # Set the time guesses as the initial values for the time-related
            # trajectory variables
            target_prob.set_val(
                parent_prefix + f'traj.{phase_name}.t_initial', t_initial, units='s'
            )
            target_prob.set_val(
                parent_prefix + f'traj.{phase_name}.t_duration', t_duration, units='s'
            )

        if 'distance' not in guesses:
            # Determine initial distance guesses depending on the phase name
            if 'desc1' == base_phase:
                ys = [aviary_group.target_range * 0.97, aviary_group.target_range * 0.99]
            elif 'desc2' in base_phase:
                ys = [aviary_group.target_range * 0.99, aviary_group.target_range]
            # Set the distance guesses as the initial values for the distance state
            # variable
            target_prob.set_val(
                parent_prefix + f'traj.{phase_name}.states:distance',
                phase.interp(Dynamic.Mission.DISTANCE, ys=ys),
            )
