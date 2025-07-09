import openmdao.api as om

from aviary.mission.flops_based.ode.range_rate import RangeRate
from aviary.mission.flops_based.ode.required_thrust import RequiredThrust
from aviary.mission.ode.altitude_rate import AltitudeRate
from aviary.mission.ode.specific_energy_rate import SpecificEnergyRate
from aviary.variable_info.variables import Dynamic

from aviary.utils.functions import add_aviary_input, add_aviary_output


class SustainedLoadFactor(om.Group):
    """Calculates instantaneous sustained load factor (no loss of energy)."""

    def initialize(self):
        self.options.declare(
            'num_nodes', types=int, desc='Number of nodes to be evaluated in the RHS'
        )

    def setup(self):
        num_nodes = self.options['num_nodes']
