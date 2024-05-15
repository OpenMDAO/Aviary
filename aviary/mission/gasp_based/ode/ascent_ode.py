import numpy as np
from dymos.models.atmosphere.atmos_1976 import USatm1976Comp

from aviary.variable_info.enums import AlphaModes, AnalysisScheme
from aviary.variable_info.variables import Aircraft, Mission, Dynamic
from aviary.mission.gasp_based.ode.ascent_eom import AscentEOM

from aviary.mission.gasp_based.ode.base_ode import BaseODE
from aviary.mission.gasp_based.ode.params import ParamPort
from aviary.variable_info.enums import AlphaModes
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


class AscentODE(BaseODE):
    """ODE for initial ascent.

    This phase is intended to model the portion of aircraft flight starting when the
    aircraft detaches from the runway, and it is capable of retracting flaps and landing
    gear as necessary.
    """

    def initialize(self):
        super().initialize()
        self.options.declare("alpha_mode", default=AlphaModes.DEFAULT, types=AlphaModes)

    def setup(self):
        nn = self.options["num_nodes"]
        alpha_mode = self.options["alpha_mode"]
        analysis_scheme = self.options["analysis_scheme"]
        aviary_options = self.options['aviary_options']
        core_subsystems = self.options['core_subsystems']

        # TODO: paramport
        ascent_params = ParamPort()
        if analysis_scheme is AnalysisScheme.SHOOTING:
            ascent_params.add_params({
                Aircraft.Design.MAX_FUSELAGE_PITCH_ANGLE: dict(units='deg', val=0),
            })
        self.add_subsystem("params", ascent_params, promotes=["*"])

        self.add_subsystem(
            "USatm",
            USatm1976Comp(
                num_nodes=nn),
            promotes_inputs=[
                ("h",
                 Dynamic.Mission.ALTITUDE)],
            promotes_outputs=[
                "rho",
                ("sos",
                 Dynamic.Mission.SPEED_OF_SOUND),
                ("temp",
                 Dynamic.Mission.TEMPERATURE),
                ("pres",
                 Dynamic.Mission.STATIC_PRESSURE),
                "viscosity"],
        )

        self.add_flight_conditions(nn)

        kwargs = {'num_nodes': nn, 'aviary_inputs': aviary_options,
                  'method': 'low_speed', 'retract_gear': True, 'retract_flaps': True}
        for subsystem in core_subsystems:
            system = subsystem.build_mission(**kwargs)
            if system is not None:
                self.add_subsystem(subsystem.name,
                                   system,
                                   promotes_inputs=subsystem.mission_inputs(**kwargs),
                                   promotes_outputs=subsystem.mission_outputs(**kwargs))

        if alpha_mode is AlphaModes.DEFAULT:
            # alpha as input
            pass
        else:
            self.AddAlphaControl(alpha_mode=alpha_mode, num_nodes=nn)

        if analysis_scheme is AnalysisScheme.SHOOTING:
            shooting_inputs = [Dynamic.Mission.DISTANCE]
        else:
            shooting_inputs = []

        self.add_subsystem(
            "ascent_eom",
            AscentEOM(num_nodes=nn,
                      analysis_scheme=analysis_scheme),
            promotes_inputs=[
                Dynamic.Mission.MASS,
                Dynamic.Mission.THRUST_TOTAL,
                Dynamic.Mission.LIFT,
                Dynamic.Mission.DRAG,
                Dynamic.Mission.VELOCITY,
                Dynamic.Mission.FLIGHT_PATH_ANGLE,
                "alpha",
            ]
            + shooting_inputs
            + ["aircraft:*"],
            promotes_outputs=[
                Dynamic.Mission.VELOCITY_RATE,
                Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE,
                Dynamic.Mission.ALTITUDE_RATE,
                Dynamic.Mission.DISTANCE_RATE,
                "alpha_rate",
                "normal_force",
                "fuselage_pitch",
                "load_factor",
            ],
        )

        if True:
            from aviary.utils.functions import create_printcomp
            dummy_comp = create_printcomp(
                all_inputs=[
                    Dynamic.Mission.DISTANCE,
                    Dynamic.Mission.DISTANCE_RATE,
                    Dynamic.Mission.THROTTLE,
                    Dynamic.Mission.THRUST_TOTAL,
                    # 'required_thrust',
                    Dynamic.Mission.ALTITUDE,
                    Dynamic.Mission.ALTITUDE_RATE,
                    'load_factor',
                    # 'required_lift',
                    Dynamic.Mission.MASS,
                    Dynamic.Mission.LIFT,
                    Dynamic.Mission.DRAG,
                    Dynamic.Mission.VELOCITY,
                    Dynamic.Mission.VELOCITY_RATE,
                    Dynamic.Mission.FLIGHT_PATH_ANGLE,
                    Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE,
                    'alpha',
                    "alpha_rate",
                    'fuselage_pitch',
                ],
                input_units={
                    'required_thrust': 'lbf',
                    'required_lift': 'lbf',
                    'alpha': 'deg',
                    'alpha_rate': 'deg/s',
                    'fuselage_pitch': 'deg',
                    Dynamic.Mission.FLIGHT_PATH_ANGLE: 'deg',
                })
            self.add_subsystem(
                "dummy_comp",
                dummy_comp(),
                promotes_inputs=["*"],)
            self.set_input_defaults(Dynamic.Mission.DISTANCE, val=0, units='NM')
            self.set_input_defaults(Dynamic.Mission.MASS, val=0, units='lbm')
            self.set_input_defaults('throttle', val=1, units='unitless')

        self.add_excess_rate_comps(nn)

        ParamPort.set_default_vals(self)
        self.set_input_defaults("t_init_flaps", val=47.5)
        self.set_input_defaults("t_init_gear", val=37.3)
        self.set_input_defaults("alpha", val=np.zeros(nn), units="deg")
        self.set_input_defaults(Dynamic.Mission.FLIGHT_PATH_ANGLE,
                                val=np.zeros(nn), units="deg")
        self.set_input_defaults(Dynamic.Mission.ALTITUDE, val=np.zeros(nn), units="ft")
        self.set_input_defaults(Dynamic.Mission.VELOCITY, val=np.zeros(nn), units="kn")
        self.set_input_defaults("t_curr", val=np.zeros(nn), units="s")
        self.set_input_defaults('aero_ramps.flap_factor:final_val', val=0.)
        self.set_input_defaults('aero_ramps.gear_factor:final_val', val=0.)
        self.set_input_defaults('aero_ramps.flap_factor:initial_val', val=1.)
        self.set_input_defaults('aero_ramps.gear_factor:initial_val', val=1.)
        self.set_input_defaults(Dynamic.Mission.MASS, val=np.ones(
            nn), units='kg')  # val here is nominal
