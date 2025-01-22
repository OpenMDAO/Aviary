import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM, RHO_SEA_LEVEL_ENGLISH
from aviary.interface.default_phase_info.two_dof_fiti import add_default_sgm_args

from aviary.mission.gasp_based.idle_descent_estimation import add_descent_estimation_as_submodel
from aviary.mission.gasp_based.ode.landing_ode import LandingSegment
from aviary.mission.gasp_based.ode.params import ParamPort
from aviary.mission.gasp_based.ode.taxi_ode import TaxiSegment
from aviary.mission.gasp_based.phases.groundroll_phase import GroundrollPhase
from aviary.mission.gasp_based.phases.rotation_phase import RotationPhase
from aviary.mission.gasp_based.phases.climb_phase import ClimbPhase
from aviary.mission.gasp_based.phases.cruise_phase import CruisePhase
from aviary.mission.gasp_based.phases.accel_phase import AccelPhase
from aviary.mission.gasp_based.phases.ascent_phase import AscentPhase
from aviary.mission.gasp_based.phases.descent_phase import DescentPhase

from aviary.utils.functions import create_opts2vals, add_opts2vals, wrapped_convert_units
from aviary.utils.process_input_decks import initialization_guessing
from aviary.variable_info.enums import AnalysisScheme
from aviary.variable_info.variables import Aircraft, Mission, Dynamic, Settings


class ProblemBuilder2DOF():
    """
    A 2DOF specific builder that customizes AviaryProblem() for use with
     two degree of freedom phases.
    """

    def initial_guesses(self, prob):
        # TODO: This should probably be moved to the set_initial_guesses() method in AviaryProblem class
        # Defines how the problem should build it's initial guesses for load_inputs()
        # this modifies mass_method, initialization_guesses, and aviary_values

        aviary_inputs = prob.aviary_inputs

        prob.initialization_guesses = initialization_guessing(
            aviary_inputs, prob.initialization_guesses, prob.engine_builders)

        aviary_inputs.set_val(
            Mission.Summary.CRUISE_MASS_FINAL,
            val=prob.initialization_guesses['cruise_mass_final'],
            units='lbm',
        )
        aviary_inputs.set_val(
            Mission.Summary.GROSS_MASS,
            val=prob.initialization_guesses['actual_takeoff_mass'],
            units='lbm',
        )

        # Deal with missing defaults in phase info:
        if prob.pre_mission_info is None:
            prob.pre_mission_info = {'include_takeoff': True,
                                     'external_subsystems': []}
        if prob.post_mission_info is None:
            prob.post_mission_info = {'include_landing': True,
                                      'external_subsystems': []}

        # Commonly referenced values
        prob.cruise_alt = aviary_inputs.get_val(
            Mission.Design.CRUISE_ALTITUDE, units='ft')
        prob.mass_defect = aviary_inputs.get_val('mass_defect', units='lbm')

        prob.cruise_mass_final = aviary_inputs.get_val(
            Mission.Summary.CRUISE_MASS_FINAL, units='lbm')

        if 'target_range' in prob.post_mission_info:
            prob.target_range = wrapped_convert_units(
                prob.post_mission_info['post_mission']['target_range'], 'NM')
            aviary_inputs.set_val(Mission.Summary.RANGE,
                                       prob.target_range, units='NM')
        else:
            prob.target_range = aviary_inputs.get_val(
                Mission.Design.RANGE, units='NM')
            aviary_inputs.set_val(Mission.Summary.RANGE, aviary_inputs.get_val(
                Mission.Design.RANGE, units='NM'), units='NM')
        prob.cruise_mach = aviary_inputs.get_val(Mission.Design.MACH)
        prob.require_range_residual = True

    def phase_info_default_location(self, prob):
        # Set the location of the default phase info for the EOM if no phase_info is specified

        if prob.analysis_scheme is AnalysisScheme.COLLOCATION:
            from aviary.interface.default_phase_info.two_dof import phase_info
        elif prob.analysis_scheme is AnalysisScheme.SHOOTING:
            from aviary.interface.default_phase_info.two_dof_fiti import phase_info, \
                phase_info_parameterization
            phase_info, _ = phase_info_parameterization(
                phase_info, None, prob.aviary_inputs)

        return phase_info

    def add_takeoff_systems(self, prob):
        # Create options to values
        OptionsToValues = create_opts2vals(
            [Aircraft.CrewPayload.NUM_PASSENGERS,
             Mission.Design.CRUISE_ALTITUDE, ])

        add_opts2vals(prob.model, OptionsToValues, prob.aviary_inputs)

        if prob.analysis_scheme is AnalysisScheme.SHOOTING:
            prob._add_fuel_reserve_component(
                post_mission=False, reserves_name='reserve_fuel_estimate')
            add_default_sgm_args(prob.descent_phases, prob.ode_args)
            add_descent_estimation_as_submodel(
                prob,
                phases=prob.descent_phases,
                cruise_mach=prob.cruise_mach,
                cruise_alt=prob.cruise_alt,
                reserve_fuel='reserve_fuel_estimate',
                all_subsystems=prob._get_all_subsystems(),
            )

        # Add thrust-to-weight ratio subsystem
        prob.model.add_subsystem(
            'tw_ratio',
            om.ExecComp(
                f'TW_ratio = Fn_SLS / (takeoff_mass * {GRAV_ENGLISH_LBM})',
                TW_ratio={'units': "unitless"},
                Fn_SLS={'units': 'lbf'},
                takeoff_mass={'units': 'lbm'},
            ),
            promotes_inputs=[('Fn_SLS', Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST),
                             ('takeoff_mass', Mission.Summary.GROSS_MASS)],
            promotes_outputs=[('TW_ratio', Aircraft.Design.THRUST_TO_WEIGHT_RATIO)],
        )

        prob.cruise_alt = prob.aviary_inputs.get_val(
            Mission.Design.CRUISE_ALTITUDE, units='ft')

        if prob.analysis_scheme is AnalysisScheme.COLLOCATION:
            # Add event transformation subsystem
            prob.model.add_subsystem(
                "event_xform",
                om.ExecComp(
                    ["t_init_gear=m*tau_gear+b", "t_init_flaps=m*tau_flaps+b"],
                    t_init_gear={"units": "s"},  # initial time that gear comes up
                    t_init_flaps={"units": "s"},  # initial time that flaps retract
                    tau_gear={"units": "unitless"},
                    tau_flaps={"units": "unitless"},
                    m={"units": "s"},
                    b={"units": "s"},
                ),
                promotes_inputs=[
                    "tau_gear",  # design var
                    "tau_flaps",  # design var
                    ("m", Mission.Takeoff.ASCENT_DURATION),
                    ("b", Mission.Takeoff.ASCENT_T_INTIIAL),
                ],
                promotes_outputs=["t_init_gear", "t_init_flaps"],  # link to h_fit
            )

        # Add taxi subsystem
        prob.model.add_subsystem(
            "taxi", TaxiSegment(**(prob.ode_args)),
            promotes_inputs=['aircraft:*', 'mission:*'],
        )

        # Calculate speed at which to initiate rotation
        prob.model.add_subsystem(
            "vrot",
            om.ExecComp(
                "Vrot = ((2 * mass * g) / (rho * wing_area * CLmax))**0.5 + dV1 + dVR",
                Vrot={"units": "ft/s"},
                mass={"units": "lbm"},
                CLmax={"units": "unitless"},
                g={"units": "lbf/lbm", "val": GRAV_ENGLISH_LBM},
                rho={"units": "slug/ft**3", "val": RHO_SEA_LEVEL_ENGLISH},
                wing_area={"units": "ft**2"},
                dV1={
                    "units": "ft/s",
                    "desc": "Increment of engine failure decision speed above stall",
                },
                dVR={
                    "units": "ft/s",
                    "desc": "Increment of takeoff rotation speed above engine failure "
                    "decision speed",
                },
            ),
            promotes_inputs=[
                ("wing_area", Aircraft.Wing.AREA),
                ("dV1", Mission.Takeoff.DECISION_SPEED_INCREMENT),
                ("dVR", Mission.Takeoff.ROTATION_SPEED_INCREMENT),
                ("CLmax", Mission.Takeoff.LIFT_COEFFICIENT_MAX),
            ],
            promotes_outputs=[('Vrot', Mission.Takeoff.ROTATION_VELOCITY)]
        )

    def add_post_mission_takeoff_systems(self, prob):
        pass

    def get_phase_builder(self, prob, phase_name, phase_options):

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
                'include one of: groundroll, rotation, accel, ascent, climb, cruise, or desc')

        return phase_builder

    def set_phase_options(self, prob, phase_name, phase_idx, phase, user_options):

        try:
            fix_initial = user_options.get_val('fix_initial')
        except KeyError:
            fix_initial = False

        try:
            fix_duration = user_options.get_val('fix_duration')
        except KeyError:
            fix_duration = False

        if 'ascent' in phase_name:
            phase.set_time_options(
                units="s",
                targets="t_curr",
                input_initial=True,
                input_duration=True,
            )

        elif 'cruise' in phase_name:
            # Time here is really the independent variable through which we are integrating.
            # In the case of the Breguet Range ODE, it's mass.
            # We rely on mass being monotonically non-increasing across the phase.
            phase.set_time_options(
                name='mass',
                fix_initial=False,
                fix_duration=False,
                units="lbm",
                targets="mass",
                initial_bounds=(0., 1.e7),
                initial_ref=100.e3,
                duration_bounds=(-1.e7, -1),
                duration_ref=50000,
            )

        elif 'descent' in phase_name:
            duration_ref = user_options.get_val("duration_ref", 's')
            phase.set_time_options(
                duration_bounds=duration_bounds,
                fix_initial=fix_initial,
                input_initial=input_initial,
                units="s",
                duration_ref=duration_ref,
            )

        else:

            input_initial = False
            time_units = phase.time_options['units']

            # Make a good guess for a reasonable intitial time scaler.
            try:
                initial_bounds = user_options.get_val('initial_bounds', units=time_units)
            except KeyError:
                initial_bounds = (None, None)

            if initial_bounds[0] is not None and initial_bounds[1] != 0.0:
                # Upper bound is good for a ref.
                user_options.set_val('initial_ref', initial_bounds[1],
                                     units=time_units)
            else:
                user_options.set_val('initial_ref', 600., time_units)

            duration_bounds = user_options.get_val("duration_bounds", time_units)
            user_options.set_val(
                'duration_ref', (duration_bounds[0] + duration_bounds[1]) / 2.,
                time_units
            )
            if phase_idx > 0:
                input_initial = True

            if fix_initial or input_initial:

                if prob.comm.size > 1:
                    # Phases are disconnected to run in parallel, so initial ref is
                    # valid.
                    initial_ref = user_options.get_val("initial_ref", time_units)
                else:
                    # Redundant on a fixed input; raises a warning if specified.
                    initial_ref = None

                phase.set_time_options(
                    fix_initial=fix_initial, fix_duration=fix_duration, units=time_units,
                    duration_bounds=user_options.get_val("duration_bounds", time_units),
                    duration_ref=user_options.get_val("duration_ref", time_units),
                    initial_ref=initial_ref,
                )

            else:  # TODO: figure out how to handle this now that fix_initial is dict
                phase.set_time_options(
                    fix_initial=fix_initial, fix_duration=fix_duration, units=time_units,
                    duration_bounds=user_options.get_val("duration_bounds", time_units),
                    duration_ref=user_options.get_val("duration_ref", time_units),
                    initial_bounds=initial_bounds,
                    initial_ref=user_options.get_val("initial_ref", time_units),
                )

        if 'cruise' not in phase_name:
            phase.add_control(
                Dynamic.Vehicle.Propulsion.THROTTLE,
                targets=Dynamic.Vehicle.Propulsion.THROTTLE,
                units='unitless',
                opt=False,
            )

    def link_phases(self, prob, phases, direct_links=True):

        if prob.analysis_scheme is AnalysisScheme.COLLOCATION:
            for ii in range(len(phases) - 1):
                phase1, phase2 = phases[ii:ii + 2]
                analytic1 = prob.phase_info[phase1]['user_options']['analytic']
                analytic2 = prob.phase_info[phase2]['user_options']['analytic']

                if not (analytic1 or analytic2):
                    # we always want time, distance, and mass to be continuous
                    states_to_link = {
                        'time': direct_links,
                        Dynamic.Mission.DISTANCE: direct_links,
                        Dynamic.Vehicle.MASS: False,
                    }

                    # if both phases are reserve phases or neither is a reserve phase
                    # (we are not on the boundary between the regular and reserve missions)
                    # and neither phase is ground roll or rotation (altitude isn't a state):
                    # we want altitude to be continous as well
                    if ((phase1 in prob.reserve_phases) == (phase2 in prob.reserve_phases)) and \
                            not ({"groundroll", "rotation"} & {phase1, phase2}) and \
                            not ('accel', 'climb1') == (phase1, phase2):  # required for convergence of FwGm
                        states_to_link[Dynamic.Mission.ALTITUDE] = direct_links

                    # if either phase is rotation, we need to connect velocity
                    # ascent to accel also requires velocity
                    if 'rotation' in (
                            phase1, phase2) or (
                            'ascent', 'accel') == (
                            phase1, phase2):
                        states_to_link[Dynamic.Mission.VELOCITY] = direct_links
                        # if the first phase is rotation, we also need alpha
                        if phase1 == 'rotation':
                            states_to_link['alpha'] = False

                    for state, connected in states_to_link.items():
                        # in initial guesses, all of the states, other than time use
                        # the same name
                        initial_guesses1 = prob.phase_info[phase1]['initial_guesses']
                        initial_guesses2 = prob.phase_info[phase2]['initial_guesses']

                        # if a state is in the initial guesses, get the units of the
                        # initial guess
                        kwargs = {}
                        if not connected:
                            if state in initial_guesses1:
                                kwargs = {'units': initial_guesses1[state][-1]}
                            elif state in initial_guesses2:
                                kwargs = {'units': initial_guesses2[state][-1]}

                        prob.traj.link_phases(
                            [phase1, phase2], [state], connected=connected, **kwargs)

                # if either phase is analytic we have to use a linkage_constraint
                else:
                    # analytic phases use the prefix "initial" for time and distance,
                    # but not mass
                    if analytic2:
                        prefix = 'initial_'
                    else:
                        prefix = ''

                    prob.traj.add_linkage_constraint(
                        phase1, phase2, 'time', prefix + 'time', connected=True)
                    prob.traj.add_linkage_constraint(
                        phase1, phase2, 'distance', prefix + 'distance',
                        connected=True)
                    prob.traj.add_linkage_constraint(
                        phase1, phase2, 'mass', 'mass', connected=False, ref=1.0e5)

            # add all params and promote them to prob.model level
            ParamPort.promote_params(
                prob.model,
                trajs=["traj"],
                phases=[
                    [*prob.regular_phases,
                     *prob.reserve_phases]
                ],
            )

            prob.model.promotes(
                "traj",
                inputs=[
                    ("ascent.parameters:t_init_gear", "t_init_gear"),
                    ("ascent.parameters:t_init_flaps", "t_init_flaps"),
                    ("ascent.t_initial", Mission.Takeoff.ASCENT_T_INTIIAL),
                    ("ascent.t_duration", Mission.Takeoff.ASCENT_DURATION),
                ],
            )

            # imitate input_initial for taxi -> groundroll
            eq = prob.model.add_subsystem(
                "taxi_groundroll_mass_constraint", om.EQConstraintComp())
            eq.add_eq_output("mass", eq_units="lbm", normalize=False,
                             ref=10000., add_constraint=True)
            prob.model.connect(
                "taxi.mass", "taxi_groundroll_mass_constraint.rhs:mass")
            prob.model.connect(
                "traj.groundroll.states:mass",
                "taxi_groundroll_mass_constraint.lhs:mass",
                src_indices=[0],
                flat_src_indices=True,
            )

            prob.model.connect("traj.ascent.timeseries.time", "h_fit.time_cp")
            prob.model.connect(
                "traj.ascent.timeseries.altitude", "h_fit.h_cp")

            prob.model.connect(f'traj.{prob.regular_phases[-1]}.states:mass',
                               Mission.Landing.TOUCHDOWN_MASS, src_indices=[-1])

            connect_map = {
                f"traj.{prob.regular_phases[-1]}.timeseries.distance": 'actual_range',
            }

        else:
            connect_map = {
                "taxi.mass": "traj.mass_initial",
                Mission.Takeoff.ROTATION_VELOCITY: "traj.SGMGroundroll_velocity_trigger",
                "traj.distance_final": 'actual_range',
                "traj.mass_final": Mission.Landing.TOUCHDOWN_MASS,
            }

        # promote all ParamPort inputs for analytic segments as well
        param_list = list(ParamPort.param_data)
        prob.model.promotes("taxi", inputs=param_list)
        prob.model.promotes("landing", inputs=param_list)
        if prob.analysis_scheme is AnalysisScheme.SHOOTING:
            param_list.append(Aircraft.Design.MAX_FUSELAGE_PITCH_ANGLE)
            prob.model.promotes("traj", inputs=param_list)
            # prob.model.list_inputs()
            # prob.model.promotes("traj", inputs=['ascent.ODE_group.eoms.'+Aircraft.Design.MAX_FUSELAGE_PITCH_ANGLE])

        prob.model.connect("taxi.mass", "vrot.mass")

        for source, target in connect_map.items():
            prob.model.connect(
                source,
                target,
                src_indices=[-1],
                flat_src_indices=True,
            )

    def add_landing_systems(self, prob):

        prob.model.add_subsystem(
            "landing",
            LandingSegment(
                **(prob.ode_args)),
            promotes_inputs=['aircraft:*', 'mission:*',
                             (Dynamic.Vehicle.MASS, Mission.Landing.TOUCHDOWN_MASS)],
            promotes_outputs=['mission:*'],
        )

        prob.model.connect(
            'pre_mission.interference_independent_of_shielded_area',
            'landing.interference_independent_of_shielded_area')
        prob.model.connect(
            'pre_mission.drag_loss_due_to_shielded_wing_area',
            'landing.drag_loss_due_to_shielded_wing_area')

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
        # Set initial guesses for the rotation mass and flight duration
        rotation_mass = prob.initialization_guesses['rotation_mass']
        flight_duration = prob.initialization_guesses['flight_duration']

        control_keys = ["velocity_rate", "throttle"]
        state_keys = [
            "altitude",
            "mass",
            Dynamic.Mission.DISTANCE,
            Dynamic.Mission.VELOCITY,
            "flight_path_angle",
            "alpha",
        ]

        if phase_name == 'ascent':
            # Alpha is a control for ascent.
            control_keys.append('alpha')

        prob_keys = ["tau_gear", "tau_flaps"]

        for guess_key, guess_data in guesses.items():
            val, units = guess_data

            # Set initial guess for time variables
            if 'time' == guess_key:
                target_prob.set_val(
                    parent_prefix + f'traj.{phase_name}.t_initial',
                    val[0],
                    units=units
                )
                target_prob.set_val(
                    parent_prefix + f'traj.{phase_name}.t_duration',
                    val[1],
                    units=units
                )

            else:
                # Set initial guess for control variables
                if guess_key in control_keys:
                    try:
                        target_prob.set_val(
                            parent_prefix + f'traj.{phase_name}.controls:{guess_key}',
                            prob._process_guess_var(val, guess_key, phase),
                            units=units
                        )

                    except KeyError:
                        try:
                            target_prob.set_val(
                                parent_prefix +
                                f'traj.{phase_name}.polynomial_controls:{guess_key}',
                                prob._process_guess_var(val, guess_key, phase),
                                units=units
                            )

                        except KeyError:
                            target_prob.set_val(
                                parent_prefix + f'traj.{phase_name}.bspline_controls:',
                                {guess_key},
                                prob._process_guess_var(val, guess_key, phase),
                                units=units
                            )

                if guess_key in control_keys:
                    pass

                # Set initial guess for state variables
                elif guess_key in state_keys:
                    target_prob.set_val(
                        parent_prefix + f'traj.{phase_name}.states:{guess_key}',
                        prob._process_guess_var(val, guess_key, phase),
                        units=units
                    )

                elif guess_key in prob_keys:
                    target_prob.set_val(parent_prefix + guess_key, val, units=units)

                elif ":" in guess_key:
                    target_prob.set_val(
                        parent_prefix +
                        f'traj.{phase_name}.{guess_key}',
                        prob._process_guess_var(
                            val,
                            guess_key,
                            phase),
                        units=units)
                else:
                    # raise error if the guess key is not recognized
                    raise ValueError(
                        f"Initial guess key {guess_key} in {phase_name} is not "
                        "recognized."
                    )

        # We need some special logic for these following variables because GASP computes
        # initial guesses using some knowledge of the mission duration and other variables
        # that are only available after calling `create_vehicle`. Thus these initial guess
        # values are not included in the `phase_info` object.
        base_phase = phase_name.removeprefix('reserve_')

        if 'mass' not in guesses:
            # Determine a mass guess depending on the phase name
            if base_phase in ["groundroll", "rotation", "ascent", "accel", "climb1"]:
                mass_guess = rotation_mass
            elif base_phase == "climb2":
                mass_guess = 0.99 * rotation_mass
            elif "desc" in base_phase:
                mass_guess = 0.9 * prob.cruise_mass_final

            # Set the mass guess as the initial value for the mass state variable
            target_prob.set_val(parent_prefix + f'traj.{phase_name}.states:mass',
                                mass_guess, units='lbm')

        if 'time' not in guesses:
            # Determine initial time and duration guesses depending on the phase name
            if 'desc1' == base_phase:
                t_initial = flight_duration * .9
                t_duration = flight_duration * .04
            elif 'desc2' in base_phase:
                t_initial = flight_duration * .94
                t_duration = 5000

            # Set the time guesses as the initial values for the time-related
            # trajectory variables
            target_prob.set_val(parent_prefix + f"traj.{phase_name}.t_initial",
                                t_initial, units='s')
            target_prob.set_val(parent_prefix + f"traj.{phase_name}.t_duration",
                                t_duration, units='s')

        if 'distance' not in guesses:
            # Determine initial distance guesses depending on the phase name
            if 'desc1' == base_phase:
                ys = [prob.target_range * .97, prob.target_range * .99]
            elif 'desc2' in base_phase:
                ys = [prob.target_range * .99, prob.target_range]
            # Set the distance guesses as the initial values for the distance state
            # variable
            target_prob.set_val(
                parent_prefix + f"traj.{phase_name}.states:distance",
                phase.interp(Dynamic.Mission.DISTANCE, ys=ys),
            )

