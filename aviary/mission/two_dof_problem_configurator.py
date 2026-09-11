import openmdao.api as om

from aviary.mission.two_dof.ode.landing_ode import LandingSegment
from aviary.mission.two_dof.ode.taxi_ode import TaxiSegment
from aviary.mission.two_dof.phases.accel_phase import AccelPhase
from aviary.mission.two_dof.phases.breguet_cruise_phase import (
    BreguetCruisePhase,
    ElectricCruisePhase,
)
from aviary.mission.two_dof.phases.flight_phase import FlightPhase
from aviary.mission.two_dof.phases.simple_cruise_phase import SimpleCruisePhase
from aviary.mission.two_dof.phases.takeoff_phase import TakeoffPhase
from aviary.mission.two_dof.polynomial_fit import PolynomialFit
from aviary.mission.problem_configurator import ProblemConfiguratorBase
from aviary.mission.utils import process_guess_var
from aviary.utils.process_input_decks import update_GASP_options
from aviary.utils.utils import wrapped_convert_units
from aviary.variable_info.enums import LegacyCode, PhaseType
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


class TwoDOFProblemConfigurator(ProblemConfiguratorBase):
    """
    A 2DOF specific builder that customizes AviaryProblem() for use with
     two-degrees-of-freedom phases.
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
        aviary_inputs = aviary_group.aviary_inputs

        aviary_inputs = update_GASP_options(aviary_inputs)

        aviary_inputs.set_val(
            Mission.GROSS_MASS,
            val=aviary_group.initialization_guesses['actual_takeoff_mass'],
            units='lbm',
        )

        # Deal with missing defaults in phase info:
        aviary_group.pre_mission_info.setdefault('include_takeoff', True)

        aviary_group.post_mission_info.setdefault('include_landing', True)

        # Commonly referenced values
        aviary_group.cruise_alt = aviary_inputs.get_val(Aircraft.Design.CRUISE_ALTITUDE, units='ft')
        aviary_group.mass_defect = aviary_inputs.get_val('mass_defect', units='lbm')

        aviary_group.cruise_mass_final = aviary_group.initialization_guesses['cruise_mass_final']

        if 'target_range' in aviary_group.post_mission_info:
            aviary_group.target_range = wrapped_convert_units(
                aviary_group.post_mission_info['target_range'], 'NM'
            )
            aviary_inputs.set_val(Mission.RANGE, aviary_group.target_range, units='NM')
        else:
            aviary_group.target_range = aviary_inputs.get_val(Aircraft.Design.RANGE, units='NM')
            aviary_inputs.set_val(
                Mission.RANGE,
                aviary_inputs.get_val(Aircraft.Design.RANGE, units='NM'),
                units='NM',
            )

        aviary_group.cruise_mach = aviary_inputs.get_val(Aircraft.Design.MACH)
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

    def get_code_origin(self):
        """
        Return the legacy of this problem configurator.

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
            Aircraft.Design.CRUISE_ALTITUDE, units='ft'
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
        PhaseBuilder
            Phase builder for requested phase.
        """
        # TODO: Move this to aviary_group and let other EOM use them.
        if 'phase_type' in phase_options['user_options']:
            builder = phase_options['user_options']['phase_type']

            if builder is PhaseType.BREGUET_RANGE:
                # TODO: This is the only one currently accessed by name string, to preserve
                # some legacy tests.
                if 'electric_cruise' in phase_name:
                    phase_builder = ElectricCruisePhase
                else:
                    phase_builder = BreguetCruisePhase

            elif builder is PhaseType.SIMPLE_CRUISE:
                phase_builder = SimpleCruisePhase

            elif builder is PhaseType.TWO_DOF_TAKEOFF:
                phase_builder = TakeoffPhase

            elif builder is PhaseType.ACCEL:
                phase_builder = AccelPhase

            else:
                phase_builder = FlightPhase

        else:
            # All other phases are flight phases.
            phase_builder = FlightPhase

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
        phase_builder = user_options['phase_type']
        if phase_builder is PhaseType.BREGUET_RANGE:
            # The Breguet Range Cruise phase integrates over mass instead of time.
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

        if initial_bounds[0] is not None and initial_bounds[1]:
            # Upper bound is good for a ref.
            time_initial_ref = initial_bounds[1]
        else:
            time_initial_ref = 600.0

        if duration_ref is None:
            # Why are we overriding this?
            duration_ref = 0.5 * (duration_bounds[0] + duration_bounds[1])

        input_initial = phase_idx > 0
        extra_args = {}

        if fix_initial or input_initial:
            if comm.size > 1:
                # Phases are disconnected to run in parallel, so initial ref is
                # valid.
                initial_ref = time_initial_ref
            else:
                # Redundant on a fixed input; raises a warning if specified.
                initial_ref = None

        else:
            extra_args['initial_bounds'] = initial_bounds

        if phase_builder is PhaseType.SIMPLE_CRUISE:
            extra_args['targets'] = 'time'
        elif phase_builder is PhaseType.TWO_DOF_TAKEOFF:
            extra_args['targets'] = 't_curr'

        phase.set_time_options(
            fix_initial=fix_initial,
            fix_duration=fix_duration,
            units=time_units,
            duration_bounds=duration_bounds,
            duration_ref=duration_ref,
            initial_ref=initial_ref,
            **extra_args,
        )

        if phase_builder is not PhaseType.SIMPLE_CRUISE:
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
        if phase_builder is PhaseType.TWO_DOF_TAKEOFF:
            # safely add in default method in way that doesn't overwrite existing method
            # and create nested structure if it doesn't already exist
            aviary_group.mission_info[phase_name].setdefault('subsystem_options', {}).setdefault(
                'aerodynamics', {}
            ).setdefault('method', 'low_speed')

    def configure_trajectory(self, aviary_group, phases):
        """
        Link or configure phase connections to other upstream or downstream components.

        This is called from AviaryProblem.link_phases

        Parameters
        ----------
        aviary_group : AviaryGroup
            Aviary model that owns this configurator.
        phases : list[Phase]
            List of all phases in the trajectory.
        """
        aviary_group.promotes(
            'traj',
            inputs=[
                ('ascent.parameters:t_init_gear', 't_init_gear'),
                ('ascent.parameters:t_init_flaps', 't_init_flaps'),
                ('ascent.t_duration', Mission.Takeoff.ASCENT_DURATION),
            ],
        )

        # Ascent t_iniitial is connected, so we need a slack constraint for the desvar that feeds
        # into the event xform.
        comp = om.ExecComp(
            ['con_val = initial_time - initial_time_slack'],
            con_val={'units': 's'},
            initial_time={'units': 's'},
            initial_time_slack={'units': 's'},
        )
        aviary_group.add_subsystem(
            'ascent_initial_time_slack_constraint',
            comp,
            promotes_inputs=[
                ('initial_time_slack', Mission.Takeoff.ASCENT_T_INITIAL),
            ],
        )
        aviary_group.connect(
            'traj.ascent.t_initial', 'ascent_initial_time_slack_constraint.initial_time'
        )
        aviary_group.add_constraint(
            'ascent_initial_time_slack_constraint.con_val',
            equals=0.0,
            ref=30.0,
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
                ('actual_range', Mission.RANGE),
                'target_range',
            ],
            promotes_outputs=[('range_resid', Mission.Constraints.RANGE_RESIDUAL)],
        )

    def _add_landing_systems(self, aviary_group):
        aviary_group.add_subsystem(
            'landing',
            LandingSegment(**(aviary_group.ode_args)),
            promotes_inputs=[
                'aircraft:*',
                'mission:*',
                (Dynamic.Vehicle.MASS, Mission.FINAL_MASS),
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
        # Breguet cruise integrates mass, so initial guesses are different.
        phase_type = aviary_group.mission_info[phase_name]['user_options']['phase_type']
        if phase_type is PhaseType.BREGUET_RANGE:
            for guess_key, guess_data in guesses.items():
                val, units = guess_data

                if 'mass' == guess_key:
                    # Set initial and duration mass for the Breguet cruise phase.
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

            # Breguet phase should have nothing else to set.
            return

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
                    target_prob.set_val(Mission.Takeoff.ASCENT_T_INITIAL, val[0], units=units)
                    target_prob.set_val(Mission.Takeoff.ASCENT_DURATION, val[1], units=units)
                else:
                    phase.set_time_val(initial=val[0], duration=val[1], units=units)

            else:
                # Set initial guess for control variables
                if guess_key in control_keys:
                    phase.set_control_val(
                        guess_key,
                        vals=process_guess_var(val, guess_key, phase),
                        units=units,
                    )

                # Set initial guess for state variables
                elif guess_key in state_keys:
                    phase.set_state_val(
                        guess_key,
                        vals=process_guess_var(val, guess_key, phase),
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
        if 'mass' not in guesses:
            # Set initial guesses for the rotation mass and flight duration
            rotation_mass = aviary_group.initialization_guesses['rotation_mass']

            # Set the mass guess as the initial value for the mass state variable
            phase.set_state_val('mass', rotation_mass, units='lbm')
