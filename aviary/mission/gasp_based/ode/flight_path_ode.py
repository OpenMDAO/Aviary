import numpy as np
import openmdao.api as om
from aviary.subsystems.mass.mass_to_weight import MassToWeight

from aviary.variable_info.enums import AlphaModes, AnalysisScheme, SpeedType
from aviary.subsystems.atmosphere.atmosphere import Atmosphere
from aviary.mission.gasp_based.ode.base_ode import BaseODE
from aviary.mission.gasp_based.ode.flight_path_eom import FlightPathEOM
from aviary.mission.gasp_based.ode.params import ParamPort
from aviary.subsystems.propulsion.propulsion_builder import PropulsionBuilderBase
from aviary.variable_info.variables import Aircraft, Dynamic, Mission
from aviary.mission.gasp_based.ode.time_integration_base_classes import add_SGM_required_inputs


class FlightPathODE(BaseODE):
    """ ODE using 2D aircraft equations of motion with states distance, alt, TAS, and gamma.

    Control is managed via angle-of-attack (alpha).
    """

    def initialize(self):
        super().initialize()
        self.options.declare("alpha_mode", default=AlphaModes.DEFAULT, types=AlphaModes)
        self.options.declare("input_speed_type", default=SpeedType.TAS, types=SpeedType,
                             desc="Whether the speed is given as a equivalent airspeed, true airspeed, or mach number")
        self.options.declare(
            "ground_roll",
            types=bool,
            default=False,
            desc="True if the aircraft is confined to the ground. Removes altitude rate as an "
            "output and adjusts the TAS rate equation.")
        self.options.declare(
            "clean",
            types=bool,
            default=False,
            desc="If true then no flaps or gear are included. Useful for high-speed flight phases.")

    def setup(self):
        self.options['auto_order'] = True
        nn = self.options["num_nodes"]
        aviary_options = self.options['aviary_options']
        alpha_mode = self.options['alpha_mode']
        analysis_scheme = self.options['analysis_scheme']
        input_speed_type = self.options['input_speed_type']

        print_level = 0

        kwargs = {'num_nodes': nn, 'aviary_inputs': aviary_options,
                  'method': 'low_speed'}
        if self.options['clean']:
            kwargs['method'] = 'cruise'
            kwargs['output_alpha'] = False

        EOM_inputs = [
            Dynamic.Mission.MASS,
            Dynamic.Mission.THRUST_TOTAL,
            Dynamic.Mission.LIFT,
            Dynamic.Mission.DRAG,
            Dynamic.Mission.VELOCITY,
            Dynamic.Mission.FLIGHT_PATH_ANGLE,
        ] + ['aircraft:*']
        if not self.options['ground_roll']:
            EOM_inputs.append('alpha')

        if analysis_scheme is AnalysisScheme.SHOOTING:
            SGM_required_inputs = {
                't_curr': {'units': 's'},
                'distance_trigger': {'units': 'ft'},
                Dynamic.Mission.ALTITUDE: {'units': 'ft'},
                Dynamic.Mission.DISTANCE: {'units': 'ft'},
            }
            if kwargs['method'] == 'cruise':
                SGM_required_inputs[Dynamic.Mission.FLIGHT_PATH_ANGLE] = {
                    'val': 0, 'units': 'deg'}
            add_SGM_required_inputs(self, SGM_required_inputs)
            prop_group = om.Group()
        else:
            prop_group = self
        core_subsystems = self.options['core_subsystems']

        # TODO: paramport
        flight_path_params = ParamPort()
        if analysis_scheme is AnalysisScheme.SHOOTING and kwargs['method'] == 'cruise':
            flight_path_params.add_params({
                Aircraft.Design.OPERATING_MASS: dict(units='lbm', val=0),
                Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS: dict(units='lbm', val=0),
                Mission.Design.RESERVE_FUEL: dict(units='lbm', val=0),
            })
        self.add_subsystem("params", flight_path_params, promotes=["*"])

        self.add_subsystem(
            name='atmosphere',
            subsys=Atmosphere(num_nodes=nn, input_speed_type=input_speed_type),
            promotes=['*'],
        )

        if alpha_mode is AlphaModes.DEFAULT:
            # alpha as input
            pass
        else:
            if alpha_mode is AlphaModes.REQUIRED_LIFT:
                self.add_subsystem(
                    "calc_weight",
                    MassToWeight(num_nodes=nn),
                    promotes_inputs=[("mass", Dynamic.Mission.MASS)],
                    promotes_outputs=["weight"]
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
                        ('thrust', Dynamic.Mission.THRUST_TOTAL),
                        'alpha',
                        ('gamma', Dynamic.Mission.FLIGHT_PATH_ANGLE),
                        ('i_wing', Aircraft.Wing.INCIDENCE)
                    ],
                    promotes_outputs=['required_lift']
                )
            self.AddAlphaControl(
                alpha_mode=alpha_mode,
                target_load_factor=1,
                atol=1e-6, rtol=1e-12, num_nodes=nn, print_level=print_level)

        for subsystem in core_subsystems:
            system = subsystem.build_mission(**kwargs)
            if system is not None:
                if isinstance(subsystem, PropulsionBuilderBase):
                    prop_group.add_subsystem(subsystem.name,
                                             system,
                                             promotes_inputs=subsystem.mission_inputs(
                                                 **kwargs),
                                             promotes_outputs=subsystem.mission_outputs(**kwargs))
                else:
                    self.add_subsystem(subsystem.name,
                                       system,
                                       promotes_inputs=subsystem.mission_inputs(
                                           **kwargs),
                                       promotes_outputs=subsystem.mission_outputs(**kwargs))

        if analysis_scheme is AnalysisScheme.SHOOTING:
            prop_group.add_subsystem(
                'calc_thrust',
                om.ExecComp(
                    # TODO fix engines not providing thrust in the right direction
                    # + weight*sin(alpha + gamma)
                    'required_thrust = ( drag  ) / cos(i_wing)',
                    required_thrust={'val': 0, 'units': 'lbf'},
                    drag={'val': 0, 'units': 'lbf'},
                    # weight={'val': 0, 'units': 'lbf'},
                    # alpha={'val': 0, 'units': 'rad'},
                    # gamma={'val': 0, 'units': 'rad'},
                    i_wing={'val': 0, 'units': 'rad'},
                ),
                promotes_inputs=[
                    ('drag', Dynamic.Mission.DRAG),
                    # 'weight',
                    # 'alpha',
                    # ('gamma', Dynamic.Mission.FLIGHT_PATH_ANGLE),
                    ('i_wing', Aircraft.Wing.INCIDENCE)
                ],
                promotes_outputs=['required_thrust']
            )

            self.AddThrottleControl(prop_group=prop_group,
                                    atol=1e-8, print_level=print_level)

        self.add_subsystem(
            "flight_path_eom",
            FlightPathEOM(
                num_nodes=nn,
                ground_roll=self.options['ground_roll'],
            ),
            promotes_inputs=EOM_inputs,
            promotes_outputs=[
                Dynamic.Mission.VELOCITY_RATE,
                Dynamic.Mission.DISTANCE_RATE,
                "normal_force",
                "fuselage_pitch",
                "load_factor",
            ],
        )

        if not self.options['ground_roll']:
            self.promotes('flight_path_eom', outputs=[
                          Dynamic.Mission.ALTITUDE_RATE, Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE])

        self.add_excess_rate_comps(nn)

        ParamPort.set_default_vals(self)
        if not self.options["clean"]:
            self.set_input_defaults("t_init_flaps", val=47.5)
            self.set_input_defaults("t_init_gear", val=37.3)
            self.set_input_defaults("t_curr", val=np.zeros(nn), units="s")
        self.set_input_defaults("alpha", val=np.zeros(nn), units="rad")
        self.set_input_defaults(Dynamic.Mission.FLIGHT_PATH_ANGLE,
                                val=np.zeros(nn), units="deg")
        self.set_input_defaults(Dynamic.Mission.ALTITUDE, val=np.zeros(nn), units="ft")
        self.set_input_defaults(Dynamic.Mission.MACH, val=np.zeros(nn), units="unitless")
        self.set_input_defaults(Dynamic.Mission.MASS, val=np.zeros(nn), units="lbm")
        self.set_input_defaults(Dynamic.Mission.VELOCITY, val=np.zeros(nn), units="kn")
