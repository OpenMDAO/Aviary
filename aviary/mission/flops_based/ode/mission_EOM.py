import openmdao.api as om

from aviary.mission.ode.altitude_rate import AltitudeRate
from aviary.mission.ode.specific_energy_rate import SpecificEnergyRate
from aviary.mission.flops_based.ode.range_rate import RangeRate
from aviary.mission.flops_based.ode.required_thrust import RequiredThrust
from aviary.variable_info.variables import Dynamic


class MissionEOM(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int,
                             desc='Number of nodes to be evaluated in the RHS')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem(
            name='required_thrust',
            subsys=RequiredThrust(num_nodes=nn),
            promotes_inputs=[
                Dynamic.Vehicle.DRAG,
                Dynamic.Mission.ALTITUDE_RATE,
                Dynamic.Atmosphere.VELOCITY,
                Dynamic.Atmosphere.VELOCITY_RATE,
                Dynamic.Vehicle.MASS,
            ],
            promotes_outputs=['thrust_required'],
        )

        self.add_subsystem(
            name='groundspeed',
            subsys=RangeRate(num_nodes=nn),
            promotes_inputs=[
                Dynamic.Mission.ALTITUDE_RATE,
                Dynamic.Atmosphere.VELOCITY,
            ],
            promotes_outputs=[Dynamic.Mission.DISTANCE_RATE],
        )

        self.add_subsystem(
            name='excess_specific_power',
            subsys=SpecificEnergyRate(num_nodes=nn),
            promotes_inputs=[
                (
                    Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
                    Dynamic.Vehicle.Propulsion.THRUST_MAX_TOTAL,
                ),
                Dynamic.Atmosphere.VELOCITY,
                Dynamic.Vehicle.MASS,
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
            name=Dynamic.Vehicle.ALTITUDE_RATE_MAX,
            subsys=AltitudeRate(num_nodes=nn),
            promotes_inputs=[
                (
                    Dynamic.Mission.SPECIFIC_ENERGY_RATE,
                    Dynamic.Mission.SPECIFIC_ENERGY_RATE_EXCESS,
                ),
                Dynamic.Atmosphere.VELOCITY_RATE,
                Dynamic.Atmosphere.VELOCITY,
            ],
            promotes_outputs=[
                (Dynamic.Mission.ALTITUDE_RATE, Dynamic.Vehicle.ALTITUDE_RATE_MAX)
            ],
        )
