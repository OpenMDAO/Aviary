import openmdao.api as om

from aviary.mission.flops_based.ode.range_rate import RangeRate
from aviary.mission.flops_based.ode.required_thrust import RequiredThrust
from aviary.mission.ode.altitude_rate import AltitudeRate
from aviary.mission.ode.specific_energy_rate import SpecificEnergyRate
from aviary.variable_info.variables import Dynamic
from aviary.constants import GRAV_METRIC_FLOPS as gravity

from aviary.utils.functions import add_aviary_input, add_aviary_output


class LoadFactor(om.ExecComp):
    """Calculates instantaneous load factor, defined as ratio of lift to weight"""

    def initialize(self):
        self.options.declare(
            'num_nodes', types=int, default=1, desc='Number of nodes to be evaluated in the RHS'
        )

    def setup(self):
        num_nodes = self.options['num_nodes']

        add_aviary_input(self, Dynamic.Vehicle.LIFT)
        add_aviary_input(self, Dynamic.Vehicle.MASS)

        self.add_output('load_factor', val=1, units='unitless')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        lift = inputs[Dynamic.Vehicle.LIFT]
        mass = inputs[Dynamic.Vehicle.MASS]

        weight = mass * gravity

        outputs['load_factor'] = lift / weight

    def compute_partials(self, inputs, J):
        lift = inputs[Dynamic.Vehicle.LIFT]
        mass = inputs[Dynamic.Vehicle.MASS]

        weight = mass * gravity

        J['load_factor', Dynamic.Vehicle.LIFT] = 1 / weight
        J['load_factor', Dynamic.Vehicle.MASS] = -lift / (gravity * mass**2)
