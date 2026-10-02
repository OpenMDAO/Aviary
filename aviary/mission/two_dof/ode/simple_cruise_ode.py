import numpy as np
import openmdao.api as om

from aviary.mission.two_dof.ode.simple_cruise_eom import DistanceComp
from aviary.mission.two_dof.ode.two_dof_ode import TwoDOFODE
from aviary.mission.ode.altitude_rate import AltitudeRate
from aviary.mission.ode.specific_energy_rate import SpecificEnergyRate
from aviary.subsystems.aerodynamics.aerodynamics_builder import AerodynamicsBuilder
from aviary.subsystems.mass.mass_to_weight import MassToWeight
from aviary.subsystems.propulsion.propulsion_builder import PropulsionBuilder
from aviary.variable_info.enums import SpeedType
from aviary.variable_info.variables import Dynamic


class SimpleCruiseODE(TwoDOFODE):
    """A simple ODE for cruise that only integrates mass."""

    def setup(self):
        nn = self.options['num_nodes']

        self.add_atmosphere(input_speed_type=SpeedType.MACH)

        self.add_subsystem(
            'calc_weight',
            MassToWeight(num_nodes=nn),
            promotes_inputs=['mass'],
            promotes_outputs=['weight'],
        )

        prop_group = self.add_subsystems_and_solver(couple_propulsion=True)

        bal = om.BalanceComp(
            name=Dynamic.Vehicle.Propulsion.THROTTLE,
            val=np.ones(nn),
            # upper=1.0,
            # lower=0.0,
            units='unitless',
            lhs_name=Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            rhs_name=Dynamic.Vehicle.DRAG,
            eq_units='lbf',
        )

        prop_group.add_subsystem(
            'thrust_balance', subsys=bal, promotes_inputs=['*'], promotes_outputs=['*']
        )

        # Preserving original options.
        prop_group.nonlinear_solver.options['rtol'] = 1e-12
        prop_group.nonlinear_solver.options['atol'] = 1e-12
        prop_group.nonlinear_solver.options['maxiter'] = 20
        prop_group.nonlinear_solver.options['err_on_non_converge'] = False
        prop_group.linear_solver = om.DirectSolver()

        # collect initial/final outputs
        self.add_subsystem(
            'distance_eom',
            DistanceComp(num_nodes=nn),
            promotes_inputs=[
                ('cruise_distance_initial', 'initial_distance'),
                ('TAS_cruise', Dynamic.Mission.VELOCITY),
                'time',
            ],
            promotes_outputs=[Dynamic.Mission.DISTANCE],
        )

        self.add_subsystem(
            name='SPECIFIC_ENERGY_RATE_EXCESS',
            subsys=SpecificEnergyRate(num_nodes=nn),
            promotes_inputs=[
                Dynamic.Mission.VELOCITY,
                Dynamic.Vehicle.MASS,
                (
                    Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
                    Dynamic.Vehicle.Propulsion.THRUST_MAX_TOTAL,
                ),
                Dynamic.Vehicle.DRAG,
            ],
            promotes_outputs=[
                (
                    Dynamic.Mission.SPECIFIC_ENERGY_RATE,
                    Dynamic.Mission.SPECIFIC_ENERGY_RATE_EXCESS,
                )
            ],
        )

        self.add_subsystem(
            name='ALTITUDE_RATE_MAX',
            subsys=AltitudeRate(num_nodes=nn),
            promotes_inputs=[
                (
                    Dynamic.Mission.SPECIFIC_ENERGY_RATE,
                    Dynamic.Mission.SPECIFIC_ENERGY_RATE_EXCESS,
                ),
                Dynamic.Mission.VELOCITY_RATE,
                Dynamic.Mission.VELOCITY,
            ],
            promotes_outputs=[(Dynamic.Mission.ALTITUDE_RATE, Dynamic.Mission.ALTITUDE_RATE_MAX)],
        )

        self.set_input_defaults(Dynamic.Mission.ALTITUDE, val=np.ones(nn), units='ft')
        self.set_input_defaults('mass', val=np.ones(nn), units='lbm')
