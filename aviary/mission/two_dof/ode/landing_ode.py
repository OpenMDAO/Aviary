import numpy as np

from aviary.mission.two_dof.ode.landing_eom import (
    GlideConditionComponent,
    LandingAltitudeComponent,
    LandingGroundRollComponent,
)
from aviary.mission.two_dof.ode.params import ParamPort
from aviary.mission.two_dof.ode.two_dof_ode import TwoDOFODE
from aviary.subsystems.aerodynamics.aerodynamics_builder import AerodyanmicsBuilder
from aviary.subsystems.atmosphere.atmosphere import Atmosphere
from aviary.subsystems.propulsion.propulsion_builder import PropulsionBuilder
from aviary.variable_info.enums import SpeedType
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


class LandingSegment(TwoDOFODE):
    """Group for a 2-degree of freedom landing ODE."""

    def setup(self):
        aviary_options = self.options['aviary_options']
        subsystems = self.options['subsystems']

        # TODO: paramport
        self.add_subsystem('params', ParamPort(), promotes=['*'])

        self.add_subsystem(
            'approach_alt_comp',
            LandingAltitudeComponent(),
            promotes_inputs=[
                Mission.Landing.OBSTACLE_HEIGHT,
                Mission.Landing.AIRPORT_ALTITUDE,
            ],
            promotes_outputs=[Mission.Landing.INITIAL_ALTITUDE],
        )

        self.add_subsystem(
            name='atmosphere',
            subsys=Atmosphere(num_nodes=1, input_speed_type=SpeedType.MACH),
            promotes_inputs=[
                (Dynamic.Mission.ALTITUDE, Mission.Landing.INITIAL_ALTITUDE),
                (Dynamic.Atmosphere.MACH, Mission.Landing.INITIAL_MACH),
            ],
            promotes_outputs=[
                Dynamic.Atmosphere.DENSITY,
                Dynamic.Atmosphere.SPEED_OF_SOUND,
                Dynamic.Atmosphere.TEMPERATURE,
                Dynamic.Atmosphere.STATIC_PRESSURE,
                Dynamic.Atmosphere.DYNAMIC_VISCOSITY,
                Dynamic.Atmosphere.DYNAMIC_PRESSURE,
            ],
        )

        # collect the propulsion group names for later use with
        for subsystem in subsystems:
            if isinstance(subsystem, AerodyanmicsBuilder):
                kwargs = {'method': 'low_speed', 'retract_flaps': True, 'retract_gear': False}
                aero_builder = subsystem
                aero_system = subsystem.build_mission(
                    num_nodes=1, aviary_inputs=aviary_options, **kwargs
                )
                self.add_subsystem(
                    subsystem.name,
                    aero_system,
                    promotes_inputs=[
                        '*',
                        (
                            Dynamic.Mission.ALTITUDE,
                            Mission.Landing.INITIAL_ALTITUDE,
                        ),
                        Dynamic.Atmosphere.DENSITY,
                        Dynamic.Atmosphere.SPEED_OF_SOUND,
                        Dynamic.Atmosphere.DYNAMIC_VISCOSITY,
                        ('airport_alt', Mission.Landing.AIRPORT_ALTITUDE),
                        (Dynamic.Atmosphere.MACH, Mission.Landing.INITIAL_MACH),
                        Dynamic.Atmosphere.DYNAMIC_PRESSURE,
                        ('flap_defl', Aircraft.Wing.FLAP_DEFLECTION_LANDING),
                        ('t_init_flaps', 't_init_flaps_app'),
                        ('t_init_gear', 't_init_gear_app'),
                        ('CL_max_flaps', Mission.Landing.LIFT_COEFFICIENT_MAX),
                        (
                            'dCL_flaps_model',
                            Mission.Landing.LIFT_COEFFICIENT_FLAP_INCREMENT,
                        ),
                        (
                            'dCD_flaps_model',
                            Mission.Landing.DRAG_COEFFICIENT_FLAP_INCREMENT,
                        ),
                    ],
                    promotes_outputs=['CL_max'],
                )

            if isinstance(subsystem, PropulsionBuilder):
                propulsion_system = subsystem.build_mission(
                    num_nodes=1, aviary_inputs=aviary_options
                )
                propulsion_mission = self.add_subsystem(
                    subsystem.name,
                    propulsion_system,
                    promotes_inputs=[
                        '*',
                        (Dynamic.Mission.ALTITUDE, Mission.Landing.AIRPORT_ALTITUDE),
                        (Dynamic.Atmosphere.MACH, Mission.Landing.INITIAL_MACH),
                    ],
                    promotes_outputs=[(Dynamic.Vehicle.Propulsion.THRUST_TOTAL, 'thrust_idle')],
                )
                propulsion_mission.set_input_defaults(Dynamic.Vehicle.Propulsion.THROTTLE, 0.0)

        self.add_subsystem(
            'glide',
            GlideConditionComponent(),
            promotes_inputs=[
                Dynamic.Atmosphere.DENSITY,
                Mission.Landing.MAXIMUM_SINK_RATE,
                Dynamic.Vehicle.MASS,
                Aircraft.Wing.AREA,
                Mission.Landing.GLIDE_TO_STALL_RATIO,
                'CL_max',
                Mission.Landing.MAXIMUM_FLARE_LOAD_FACTOR,
                Mission.Landing.TOUCHDOWN_SINK_RATE,
                Mission.Landing.INITIAL_ALTITUDE,
                Mission.Landing.BRAKING_DELAY,
            ],
            promotes_outputs=[
                Mission.Landing.INITIAL_VELOCITY,
                Mission.Landing.STALL_VELOCITY,
                'TAS_touchdown',
                'density_ratio',
                'wing_loading_land',
                'theta',
                'glide_distance',
                'tr_distance',
                'delay_distance',
                'flare_alt',
            ],
        )

        self.add_subsystem(
            name='atmosphere_td',
            subsys=Atmosphere(num_nodes=1),
            promotes_inputs=[
                (Dynamic.Mission.ALTITUDE, Mission.Landing.AIRPORT_ALTITUDE),
                (Dynamic.Mission.VELOCITY, 'TAS_touchdown'),
            ],
            promotes_outputs=[
                (Dynamic.Atmosphere.DENSITY, 'rho_td'),
                (Dynamic.Atmosphere.SPEED_OF_SOUND, 'sos_td'),
                (Dynamic.Atmosphere.TEMPERATURE, 'T_td'),
                (Dynamic.Atmosphere.DYNAMIC_VISCOSITY, 'viscosity_td'),
                (Dynamic.Atmosphere.DYNAMIC_PRESSURE, 'q_td'),
                (Dynamic.Atmosphere.MACH, 'mach_td'),
            ],
        )

        kwargs = {'method': 'low_speed', 'retract_flaps': True, 'retract_gear': False}

        self.add_subsystem(
            'aero_td',
            aero_builder.build_mission(num_nodes=1, aviary_inputs=aviary_options, **kwargs),
            promotes_inputs=[
                '*',
                (Dynamic.Mission.ALTITUDE, Mission.Landing.AIRPORT_ALTITUDE),
                (Dynamic.Atmosphere.DENSITY, 'rho_td'),
                (Dynamic.Atmosphere.SPEED_OF_SOUND, 'sos_td'),
                (Dynamic.Atmosphere.DYNAMIC_VISCOSITY, 'viscosity_td'),
                ('airport_alt', Mission.Landing.AIRPORT_ALTITUDE),
                (Dynamic.Atmosphere.MACH, 'mach_td'),
                (Dynamic.Atmosphere.DYNAMIC_PRESSURE, 'q_td'),
                (Dynamic.Vehicle.ANGLE_OF_ATTACK, Aircraft.Wing.INCIDENCE),
                ('flap_defl', Aircraft.Wing.FLAP_DEFLECTION_LANDING),
                ('CL_max_flaps', Mission.Landing.LIFT_COEFFICIENT_MAX),
                (
                    'dCL_flaps_model',
                    Mission.Landing.LIFT_COEFFICIENT_FLAP_INCREMENT,
                ),
                (
                    'dCD_flaps_model',
                    Mission.Landing.DRAG_COEFFICIENT_FLAP_INCREMENT,
                ),
                ('t_init_flaps', 't_init_flaps_td'),
                ('t_init_gear', 't_init_gear_td'),
            ],
            promotes_outputs=[('CD', 'touchdown_CD'), ('CL', 'touchdown_CL')],
        )
        # GASP seems to run groundroll with flaps up and gear down (IWLD=2)
        self.set_input_defaults('t_init_flaps_td', 1e10)  # never deploy
        self.set_input_defaults('t_init_gear_td', 1e10)  # ensure gear down

        self.add_subsystem(
            'landinggroundroll',
            LandingGroundRollComponent(),
            promotes_inputs=[
                'touchdown_CD',
                'touchdown_CL',
                'TAS_touchdown',
                'thrust_idle',
                'density_ratio',
                'wing_loading_land',
                'glide_distance',
                'tr_distance',
                'delay_distance',
                'CL_max',
                Dynamic.Vehicle.MASS,
                'mission:*',
            ],
            promotes_outputs=[
                'ground_roll_distance',
                'average_acceleration',
                'mission:*',
            ],
        )

        ParamPort.set_default_vals(self)

        self.set_input_defaults(Mission.Landing.INITIAL_MACH, val=0.1)

        # landing doesn't change flap or gear position
        self.set_input_defaults('t_init_flaps_app', val=1e10)
        self.set_input_defaults('t_init_gear_app', val=1e10)
        self.set_input_defaults(Mission.Landing.INITIAL_ALTITUDE, val=50, units='ft')
        self.set_input_defaults('aero_ramps.flap_factor:final_val', val=1.0)
        self.set_input_defaults('aero_ramps.gear_factor:final_val', val=1.0)
        self.set_input_defaults('aero_ramps.flap_factor:initial_val', val=0.0)
        self.set_input_defaults('aero_ramps.gear_factor:initial_val', val=0.0)

        self.set_input_defaults(Aircraft.Wing.AREA, val=1.0, units='ft**2')

        # Throttle Idle
        num_engine_types = len(aviary_options.get_val(Aircraft.Engine.NUM_ENGINES))
        self.set_input_defaults(
            Dynamic.Vehicle.Propulsion.THROTTLE, np.zeros((1, num_engine_types))
        )
