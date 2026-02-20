import numpy as np
import openmdao.api as om

from aviary.mission.two_dof.ode.params import ParamPort
from aviary.mission.two_dof.ode.takeoff_eom import TakeoffEOM
from aviary.mission.two_dof.ode.two_dof_ode import TwoDOFODE
from aviary.subsystems.aerodynamics.aerodynamics_builder import AerodynamicsBuilder
from aviary.subsystems.mass.mass_to_weight import MassToWeight
from aviary.subsystems.propulsion.propulsion_builder import PropulsionBuilder
from aviary.variable_info.enums import AlphaModes, SpeedType
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


class TakeOffODE(TwoDOFODE):
    """ODE using 2D aircraft equations of motion with states distance, alt, TAS, and gamma.

    Control is managed via angle-of-attack (alpha).
    This ODE is used for two-dof takeoff phases, and supports ground roll, rotation, and
    ascending flight.
    """

    def initialize(self):
        super().initialize()

        self.options.declare('alpha_mode', default=AlphaModes.DEFAULT, types=AlphaModes)

        self.options.declare(
            'input_speed_type',
            default=SpeedType.TAS,
            types=SpeedType,
            desc='Whether the speed is given as a equivalent airspeed, true airspeed, or Mach number',
        )

        self.options.declare(
            'ground_roll',
            types=bool,
            default=False,
            desc='True if the aircraft is confined to the ground. Angle of attack is fixed and '
            'removed as an input.',
        )

        self.options.declare(
            'rotation',
            types=bool,
            default=False,
            desc='True if the aircraft is pitching up, but the rear wheels are still on the '
            'ground.',
        )

        self.options.declare(
            'clean',
            types=bool,
            default=False,
            desc='If true then no flaps or gear are included. Useful for high-speed flight phases.',
        )

    def setup(self):
        nn = self.options['num_nodes']
        aviary_options = self.options['aviary_options']
        alpha_mode = self.options['alpha_mode']
        input_speed_type = self.options['input_speed_type']
        ground_roll = self.options['ground_roll']
        rotation = self.options['rotation']

        EOM_inputs = [
            Dynamic.Vehicle.MASS,
            Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            Dynamic.Vehicle.LIFT,
            Dynamic.Vehicle.DRAG,
            Dynamic.Mission.VELOCITY,
            Dynamic.Mission.FLIGHT_PATH_ANGLE,
        ] + ['aircraft:*']
        if not self.options['ground_roll']:
            EOM_inputs.append(Dynamic.Vehicle.ANGLE_OF_ATTACK)

        subsystems = self.options['subsystems']
        subsystem_options = self.options['subsystem_options']

        # TODO: paramport
        self.add_subsystem('params', ParamPort(), promotes=['*'])

        self.add_atmosphere(input_speed_type=input_speed_type)

        if ground_roll:
            # Angle of attack equals the incidence angle.
            self.add_subsystem(
                'init_alpha',
                om.ExecComp(
                    'alpha = i_wing',
                    i_wing={'units': 'deg', 'val': 1.1},
                    alpha={'units': 'deg', 'val': 1.1 * np.ones(nn)},
                ),
                promotes=[
                    ('i_wing', Aircraft.Wing.INCIDENCE),
                    ('alpha', Dynamic.Vehicle.ANGLE_OF_ATTACK),
                ],
            )

        if alpha_mode is AlphaModes.DEFAULT:
            # alpha as input
            pass
        else:
            if alpha_mode is AlphaModes.REQUIRED_LIFT:
                self.add_subsystem(
                    'calc_weight',
                    MassToWeight(num_nodes=nn),
                    promotes_inputs=[('mass', Dynamic.Vehicle.MASS)],
                    promotes_outputs=['weight'],
                )
                self.add_subsystem(
                    'calc_lift',
                    om.ExecComp(
                        'required_lift = weight*cos(alpha + gamma) - thrust*sin(i_wing)',
                        required_lift={'val': 0, 'units': 'lbf'},
                        weight={'val': 0, 'units': 'lbf'},
                        thrust={'val': 0, 'units': 'lbf'},
                        alpha={'val': 0, 'units': 'rad'},
                        gamma={'val': 0, 'units': 'rad'},
                        i_wing={'val': 0, 'units': 'rad'},
                    ),
                    promotes_inputs=[
                        'weight',
                        ('thrust', Dynamic.Vehicle.Propulsion.THRUST_TOTAL),
                        ('alpha', Dynamic.Vehicle.ANGLE_OF_ATTACK),
                        ('gamma', Dynamic.Mission.FLIGHT_PATH_ANGLE),
                        ('i_wing', Aircraft.Wing.INCIDENCE),
                    ],
                    promotes_outputs=['required_lift'],
                )
            self.add_alpha_control(
                alpha_mode=alpha_mode,
                target_load_factor=1,
                atol=1e-6,
                rtol=1e-12,
                num_nodes=nn,
                print_level=0,
            )

        for subsystem in subsystems:
            name = subsystem.name
            kwargs = {}

            if isinstance(subsystem, AerodynamicsBuilder):
                kwargs = {'method': 'low_speed'}
                if self.options['clean']:
                    kwargs['method'] = 'cruise'
                    kwargs['output_alpha'] = False

                if not (ground_roll or rotation):
                    kwargs['retract_gear'] = True
                    kwargs['retract_flaps'] = True

            if name in subsystem_options:
                kwargs.update(subsystem_options[name])

            system = subsystem.build_mission(num_nodes=nn, aviary_inputs=aviary_options, **kwargs)

            if system is not None:
                if isinstance(subsystem, PropulsionBuilder):
                    self.add_subsystem(
                        name,
                        system,
                        promotes_inputs=subsystem.mission_inputs(**kwargs),
                        promotes_outputs=subsystem.mission_outputs(**kwargs),
                    )
                else:
                    self.add_subsystem(
                        name,
                        system,
                        promotes_inputs=subsystem.mission_inputs(**kwargs),
                        promotes_outputs=subsystem.mission_outputs(**kwargs),
                    )

        if ground_roll:
            eom_name = 'groundroll_eom'
        elif rotation:
            eom_name = 'rotation_eom'
        else:
            eom_name = 'ascent_eom'

        self.add_subsystem(
            eom_name,
            TakeoffEOM(
                num_nodes=nn,
                ground_roll=ground_roll,
                rotation=rotation,
            ),
            promotes_inputs=EOM_inputs,
            promotes_outputs=[
                Dynamic.Mission.VELOCITY_RATE,
                Dynamic.Mission.DISTANCE_RATE,
                Dynamic.Mission.ALTITUDE_RATE,
                Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE,
                'normal_force',
                'fuselage_pitch',
                'load_factor',
                'angle_of_attack_rate',
            ],
        )

        if ground_roll:
            self.add_subsystem(
                'exec',
                om.ExecComp(
                    'over_a = velocity / velocity_rate',
                    velocity_rate={'units': 'kn/s', 'val': np.ones(nn)},
                    velocity={'units': 'kn', 'val': np.ones(nn)},
                    over_a={'units': 's', 'val': np.ones(nn)},
                    has_diag_partials=True,
                ),
                promotes=['*'],
            )

            self.add_subsystem(
                'exec2',
                om.ExecComp(
                    'dt_dv = 1 / velocity_rate',
                    velocity_rate={'units': 'kn/s', 'val': np.ones(nn)},
                    dt_dv={'units': 's/kn', 'val': np.ones(nn)},
                    has_diag_partials=True,
                ),
                promotes=['*'],
            )

            self.add_subsystem(
                'exec3',
                om.ExecComp(
                    'dmass_dv = mass_rate * dt_dv',
                    mass_rate={'units': 'lbm/s', 'val': np.ones(nn)},
                    dt_dv={'units': 's/kn', 'val': np.ones(nn)},
                    dmass_dv={'units': 'lbm/kn', 'val': np.ones(nn)},
                    has_diag_partials=True,
                ),
                promotes_outputs=[
                    'dmass_dv',
                ],
                promotes_inputs=[
                    ('mass_rate', Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL),
                    'dt_dv',
                ],
            )

        if not (ground_roll or rotation):
            self.add_excess_rate_comps(nn)

        ParamPort.set_default_vals(self)

        self.set_input_defaults(Dynamic.Vehicle.ANGLE_OF_ATTACK, val=np.zeros(nn), units='rad')
        self.set_input_defaults(Dynamic.Mission.FLIGHT_PATH_ANGLE, val=np.zeros(nn), units='deg')
        self.set_input_defaults(Dynamic.Mission.VELOCITY, val=np.zeros(nn), units='kn')
        self.set_input_defaults(Dynamic.Mission.ALTITUDE, val=np.zeros(nn), units='ft')
        self.set_input_defaults(Dynamic.Vehicle.MASS, val=np.zeros(nn), units='lbm')

        # TODO: Some of these are backdoor defaults.
        if not self.options['clean']:
            self.set_input_defaults('t_init_flaps', val=47.5, units='s')
            self.set_input_defaults('t_init_gear', val=37.3, units='s')
            self.set_input_defaults('t_curr', val=np.zeros(nn), units='s')
            if ground_roll or rotation:
                self.set_input_defaults('aero_ramps.flap_factor:final_val', val=1.0)
                self.set_input_defaults('aero_ramps.gear_factor:final_val', val=1.0)
            else:
                self.set_input_defaults('aero_ramps.flap_factor:final_val', val=0.0)
                self.set_input_defaults('aero_ramps.gear_factor:final_val', val=0.0)
            self.set_input_defaults('aero_ramps.flap_factor:initial_val', val=1.0)
            self.set_input_defaults('aero_ramps.gear_factor:initial_val', val=1.0)

        if ground_roll:
            self.set_input_defaults(Dynamic.Mission.VELOCITY_RATE, val=np.zeros(nn), units='kn/s')
            self.set_input_defaults(Aircraft.Wing.INCIDENCE, val=1.0, units='deg')
