import numpy as np
import openmdao.api as om

from aviary.mission.gasp_based.ode.params import ParamPort
from aviary.mission.gasp_based.ode.rotation_eom import RotationEOM
from aviary.mission.gasp_based.ode.time_integration_base_classes import add_SGM_required_inputs
from aviary.mission.gasp_based.ode.two_dof_ode import TwoDOFODE
from aviary.variable_info.enums import AnalysisScheme
from aviary.variable_info.variables import Aircraft, Dynamic


class RotationODE(TwoDOFODE):
    """ODE for takeoff rotation.

    This phase spans the time from when the aircraft is touching the runway but has
    begun to rotate to liftoff.
    """

    def setup(self):
        nn = self.options['num_nodes']
        analysis_scheme = self.options['analysis_scheme']

        if analysis_scheme is AnalysisScheme.SHOOTING:
            add_SGM_required_inputs(
                self,
                {
                    Dynamic.Mission.DISTANCE: {'units': 'ft'},
                },
            )

        # TODO: paramport
        self.add_subsystem('params', ParamPort(), promotes=['*'])

        self.add_atmosphere()

        kwargs = {'method': 'low_speed'}
        self.options['subsystem_options'].setdefault('core_aerodynamics', {}).update(kwargs)

        self.add_core_subsystems()

        self.add_external_subsystems()

        if analysis_scheme is AnalysisScheme.SHOOTING:
            alpha_comp = om.ExecComp(
                'alpha=rotation_rate*(t_curr-start_rotation)+alpha_init',
                alpha=dict(val=0.0, units='deg'),
                rotation_rate=dict(val=10.0 / 3.0, units='deg/s'),
                t_curr=dict(val=0.0, units='s'),
                start_rotation=dict(val=0.0, units='s'),
                alpha_init=dict(val=0.0, units='deg'),
            )
            alpha_comp_inputs = [
                'rotation_rate',
                't_curr',
                'start_rotation',
                ('alpha_init', Aircraft.Wing.INCIDENCE),
            ]
            self.add_subsystem(
                'alpha_comp',
                alpha_comp,
                promotes_inputs=alpha_comp_inputs,
                promotes_outputs=[('alpha', Dynamic.Vehicle.ANGLE_OF_ATTACK)],
            )

        self.add_subsystem('rotation_eom', RotationEOM(num_nodes=nn), promotes=['*'])

        ParamPort.set_default_vals(self)
        self.set_input_defaults('t_init_flaps', val=47.5, units='s')
        self.set_input_defaults('t_init_gear', val=37.3, units='s')
        self.set_input_defaults(Dynamic.Vehicle.ANGLE_OF_ATTACK, val=np.ones(nn), units='deg')
        self.set_input_defaults(Dynamic.Mission.FLIGHT_PATH_ANGLE, val=np.zeros(nn), units='deg')
        self.set_input_defaults(Dynamic.Mission.ALTITUDE, val=np.zeros(nn), units='ft')
        self.set_input_defaults(Dynamic.Mission.VELOCITY, val=np.zeros(nn), units='kn')
        self.set_input_defaults('t_curr', val=np.zeros(nn), units='s')
        self.set_input_defaults('aero_ramps.flap_factor:final_val', val=1.0)
        self.set_input_defaults('aero_ramps.gear_factor:final_val', val=1.0)
        self.set_input_defaults('aero_ramps.flap_factor:initial_val', val=1.0)
        self.set_input_defaults('aero_ramps.gear_factor:initial_val', val=1.0)
