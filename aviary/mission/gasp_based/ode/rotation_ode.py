import numpy as np
import openmdao.api as om
from dymos.models.atmosphere.atmos_1976 import USatm1976Comp

from aviary.mission.gasp_based.ode.base_ode import BaseODE
from aviary.mission.gasp_based.ode.params import ParamPort
from aviary.mission.gasp_based.ode.rotation_eom import RotationEOM
from aviary.variable_info.enums import AnalysisScheme
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


class RotationODE(BaseODE):
    """ODE for takeoff rotation.

    This phase spans the time from when the aircraft is touching the runway but has
    begun to rotate to liftoff.
    """

    def setup(self):
        nn = self.options["num_nodes"]
        analysis_scheme = self.options["analysis_scheme"]
        aviary_options = self.options['aviary_options']
        core_subsystems = self.options['core_subsystems']

        # TODO: paramport
        self.add_subsystem("params", ParamPort(), promotes=["*"])

        self.add_subsystem(
            "USatm", USatm1976Comp(
                num_nodes=nn), promotes_inputs=[
                ("h", Dynamic.Mission.ALTITUDE)], promotes_outputs=[
                "rho", ("sos", Dynamic.Mission.SPEED_OF_SOUND), ("temp", Dynamic.Mission.TEMPERATURE), ("pres", Dynamic.Mission.STATIC_PRESSURE), "viscosity"], )

        self.add_flight_conditions(nn)

        kwargs = {'num_nodes': nn, 'aviary_inputs': aviary_options,
                  'method': 'low_speed'}
        for subsystem in core_subsystems:
            system = subsystem.build_mission(**kwargs)
            if system is not None:
                self.add_subsystem(subsystem.name,
                                   system,
                                   promotes_inputs=subsystem.mission_inputs(**kwargs),
                                   promotes_outputs=subsystem.mission_outputs(**kwargs))

        if analysis_scheme is AnalysisScheme.SHOOTING:
            alpha_comp = om.ExecComp(
                'alpha=rotation_rate*(t_curr-start_rotation)+alpha_init',
                alpha=dict(val=0., units='deg'),
                rotation_rate=dict(val=10.0/3.0, units='deg/s'),
                t_curr=dict(val=0., units='s'),
                start_rotation=dict(val=0., units='s'),
                alpha_init=dict(val=0., units='deg'),
            )
            alpha_comp_inputs = ["rotation_rate", "t_curr", "start_rotation",
                                 ("alpha_init", Aircraft.Wing.INCIDENCE)]
            self.add_subsystem("alpha_comp",
                               alpha_comp,
                               promotes_inputs=alpha_comp_inputs,
                               promotes_outputs=["alpha"],
                               )

        self.add_subsystem("rotation_eom", RotationEOM(
            num_nodes=nn, analysis_scheme=analysis_scheme), promotes=["*"])

        if False:
            from aviary.utils.functions import create_printcomp
            dummy_comp = create_printcomp(
                all_inputs=[
                    Dynamic.Mission.DISTANCE,
                    Dynamic.Mission.DISTANCE_RATE,
                    Dynamic.Mission.THROTTLE,
                    Dynamic.Mission.THRUST_TOTAL,
                    'required_thrust',
                    Dynamic.Mission.ALTITUDE,
                    Dynamic.Mission.ALTITUDE_RATE,
                    'load_factor',
                    'required_lift',
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
                    'normal_force',
                    't_init_flaps',
                    't_init_gear',
                    't_curr',
                ],
                input_units={
                    'required_thrust': 'lbf',
                    'required_lift': 'lbf',
                    'alpha': 'deg',
                    Dynamic.Mission.FLIGHT_PATH_ANGLE: 'deg',
                })
            self.add_subsystem(
                "dummy_comp",
                dummy_comp(),
                promotes_inputs=["*"],)
            self.set_input_defaults(Dynamic.Mission.DISTANCE, val=0, units='NM')
            self.set_input_defaults(Dynamic.Mission.MASS, val=0, units='lbm')
            self.set_input_defaults('throttle', val=1, units='unitless')

        ParamPort.set_default_vals(self)
        self.set_input_defaults("t_init_flaps", val=47.5, units='s')
        self.set_input_defaults("t_init_gear", val=37.3, units='s')
        self.set_input_defaults("alpha", val=np.ones(nn), units="deg")
        self.set_input_defaults(Dynamic.Mission.FLIGHT_PATH_ANGLE,
                                val=np.zeros(nn), units="deg")
        self.set_input_defaults(Dynamic.Mission.ALTITUDE, val=np.zeros(nn), units="ft")
        self.set_input_defaults(Dynamic.Mission.VELOCITY, val=np.zeros(nn), units="kn")
        self.set_input_defaults("t_curr", val=np.zeros(nn), units="s")
        self.set_input_defaults('aero_ramps.flap_factor:final_val', val=1.)
        self.set_input_defaults('aero_ramps.gear_factor:final_val', val=1.)
        self.set_input_defaults('aero_ramps.flap_factor:initial_val', val=1.)
        self.set_input_defaults('aero_ramps.gear_factor:initial_val', val=1.)
