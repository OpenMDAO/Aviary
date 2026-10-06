import openmdao.api as om

from aviary.variable_info.functions import add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft


class CargoContainerMass(om.ExplicitComponent):
    """
    Calculate the mass of cargo containers. The methodology is based on the GASP weight equations,
    modified to output mass instead of weight.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_PASSENGERS)
        # keep it as an option, otherwise we will face a scenario of many stairs
        add_aviary_option(self, Aircraft.CrewPayload.ULD_MASS_PER_PASSENGER)

    def setup(self):
        add_aviary_output(self, Aircraft.CrewPayload.CARGO_CONTAINER_MASS, units='lbm')

    def compute(self, inputs, outputs):
        PAX = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]
        # Some aircraft don’t use ULD’s, like the Boeing Max-8, so it can be 0.
        uld_per_pax = self.options[Aircraft.CrewPayload.ULD_MASS_PER_PASSENGER][0]

        # mass of a single ULD (LD-3 type)
        unit_mass_cargo_handling = 165.0

        cargo_handling_mass = (int(PAX * uld_per_pax) + 1) * unit_mass_cargo_handling

        outputs[Aircraft.CrewPayload.CARGO_CONTAINER_MASS] = cargo_handling_mass
