from aviary.mission.gasp_based.ode.taxi_ode import TaxiSegment
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.functions import create_opts2vals, add_opts2vals, wrapped_convert_units
from aviary.utils.process_input_decks import initialization_guessing, update_GASP_options
from aviary.variable_info.enums import AnalysisScheme
from aviary.variable_info.variables import Aircraft, Mission, Dynamic, Settings


class AviaryProblemBuilder_2DOF():
    """
    A 2DOF specific builder that customizes AviaryProblem() for use with
     two degree of freedom phases.
    """

    def initial_guesses(self, prob, engine_builders):
        # TODO: This should probably be moved to the set_initial_guesses() method in AviaryProblem class
        # Defines how the problem should build it's initial guesses for load_inputs()
        # this modifies mass_method, initialization_guesses, and aviary_values

        aviary_inputs = prob.aviary_inputs
        prob.mass_method = aviary_inputs.get_val(Settings.MASS_METHOD)

        if engine_builders is None:
            engine_builders = build_engine_deck(aviary_inputs)
        prob.engine_builders = engine_builders

        aviary_inputs = update_GASP_options(aviary_inputs)

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

