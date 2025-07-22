import numpy as np

from aviary.mission.gasp_based.ode.ascent_eom import AscentEOM
from aviary.mission.gasp_based.ode.params import ParamPort
from aviary.mission.gasp_based.ode.two_dof_ode import TwoDOFODE
from aviary.variable_info.enums import AlphaModes
from aviary.variable_info.variables import Aircraft, Dynamic


class AscentODE(TwoDOFODE):
    """ODE for initial ascent.

    This phase is intended to model the portion of aircraft flight starting when the
    aircraft detaches from the runway, and it is capable of retracting flaps and landing
    gear as necessary.
    """

    def initialize(self):
        super().initialize()
        self.options.declare('alpha_mode', default=AlphaModes.DEFAULT, types=AlphaModes)

    def setup(self):
        nn = self.options['num_nodes']
        alpha_mode = self.options['alpha_mode']

        # TODO: paramport
        ascent_params = ParamPort()
        self.add_subsystem('params', ascent_params, promotes=['*'])

        self.add_atmosphere()

        kwargs = {
            'method': 'low_speed',
            'retract_gear': True,
            'retract_flaps': True,
        }
        self.options['subsystem_options'].setdefault('core_aerodynamics', {}).update(kwargs)

        self.add_core_subsystems()

        self.add_external_subsystems()

        if alpha_mode is AlphaModes.DEFAULT:
            # alpha as input
            pass
        else:
            self.add_alpha_control(alpha_mode=alpha_mode, num_nodes=nn)

        self.add_subsystem(
            'ascent_eom',
            AscentEOM(num_nodes=nn),
            promotes_inputs=[
                Dynamic.Vehicle.MASS,
                Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
                Dynamic.Vehicle.LIFT,
                Dynamic.Vehicle.DRAG,
                Dynamic.Mission.VELOCITY,
                Dynamic.Mission.FLIGHT_PATH_ANGLE,
                Dynamic.Vehicle.ANGLE_OF_ATTACK,
            ]
            + ['aircraft:*'],
            promotes_outputs=[
                Dynamic.Mission.VELOCITY_RATE,
                Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE,
                Dynamic.Mission.ALTITUDE_RATE,
                Dynamic.Mission.DISTANCE_RATE,
                'angle_of_attack_rate',
                'normal_force',
                'fuselage_pitch',
                'load_factor',
            ],
        )

        self.add_excess_rate_comps(nn)

        ParamPort.set_default_vals(self)
        self.set_input_defaults('t_init_flaps', val=47.5)
        self.set_input_defaults('t_init_gear', val=37.3)
        self.set_input_defaults(Dynamic.Vehicle.ANGLE_OF_ATTACK, val=np.zeros(nn), units='deg')
        self.set_input_defaults(Dynamic.Mission.FLIGHT_PATH_ANGLE, val=np.zeros(nn), units='deg')
        self.set_input_defaults(Dynamic.Mission.ALTITUDE, val=np.zeros(nn), units='ft')
        self.set_input_defaults(Dynamic.Mission.VELOCITY, val=np.zeros(nn), units='kn')
        self.set_input_defaults('t_curr', val=np.zeros(nn), units='s')
        self.set_input_defaults('aero_ramps.flap_factor:final_val', val=0.0)
        self.set_input_defaults('aero_ramps.gear_factor:final_val', val=0.0)
        self.set_input_defaults('aero_ramps.flap_factor:initial_val', val=1.0)
        self.set_input_defaults('aero_ramps.gear_factor:initial_val', val=1.0)
        self.set_input_defaults(
            Dynamic.Vehicle.MASS, val=np.ones(nn), units='kg'
        )  # val here is nominal
