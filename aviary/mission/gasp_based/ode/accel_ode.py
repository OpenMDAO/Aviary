import numpy as np

from aviary.mission.gasp_based.ode.accel_eom import AccelerationRates
from aviary.mission.gasp_based.ode.params import ParamPort
from aviary.mission.gasp_based.ode.time_integration_base_classes import (
    add_SGM_required_inputs,
    add_SGM_required_outputs,
)
from aviary.mission.gasp_based.ode.two_dof_ode import TwoDOFODE
from aviary.subsystems.mass.mass_to_weight import MassToWeight
from aviary.variable_info.enums import AnalysisScheme
from aviary.variable_info.variables import Dynamic


class AccelODE(TwoDOFODE):
    """ODE for level acceleration.

    In level acceleration, there are only nonzero net forces in the direction of motion.
    There is a balance component to solve for the angle of attack necessary to make lift
    equal to weight. Acceleration results from engine thrust in excess of drag.
    """

    def setup(self):
        nn = self.options['num_nodes']
        analysis_scheme = self.options['analysis_scheme']

        if analysis_scheme is AnalysisScheme.SHOOTING:
            add_SGM_required_inputs(
                self,
                {
                    't_curr': {'units': 's'},
                    Dynamic.Mission.DISTANCE: {'units': 'ft'},
                },
            )
            add_SGM_required_outputs(
                self,
                {
                    Dynamic.Mission.ALTITUDE_RATE: {'units': 'ft/s'},
                },
            )

        # TODO: paramport
        self.add_subsystem('params', ParamPort(), promotes=['*'])

        self.add_atmosphere()

        self.add_subsystem(
            'calc_weight',
            MassToWeight(num_nodes=nn),
            promotes_inputs=[('mass', Dynamic.Vehicle.MASS)],
            promotes_outputs=['weight'],
        )

        kwargs = {
            'method': 'cruise',
            'output_alpha': True,
        }
        self.options['subsystem_options'].setdefault('core_aerodynamics', {}).update(kwargs)

        self.add_core_subsystems()

        self.add_external_subsystems()

        self.add_subsystem(
            'accel_eom',
            AccelerationRates(num_nodes=nn),
            promotes_inputs=[
                Dynamic.Vehicle.MASS,
                Dynamic.Mission.VELOCITY,
                Dynamic.Vehicle.DRAG,
                Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            ],
            promotes_outputs=[
                Dynamic.Mission.VELOCITY_RATE,
                Dynamic.Mission.DISTANCE_RATE,
            ],
        )

        self.add_excess_rate_comps(nn)

        ParamPort.set_default_vals(self)
        self.set_input_defaults(Dynamic.Vehicle.MASS, val=14e4 * np.ones(nn), units='lbm')
        self.set_input_defaults(Dynamic.Mission.ALTITUDE, val=500 * np.ones(nn), units='ft')
        self.set_input_defaults(
            Dynamic.Mission.VELOCITY, val=200 * np.ones(nn), units='m/s'
        )  # val here is nominal
