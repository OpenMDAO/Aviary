"""
Define utilities to calculate the estimated mass of any passengers, their
baggage, and other cargo. The methodology is based on the FLOPS weight
equations, modified to output mass instead of weight.
"""

import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft


class PayloadGroup(om.Group):
    def setup(self):
        self.add_subsystem('cargo_mass', CargoMass(), promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem(
            'pax_payload_mass', PaxPayloadMass(), promotes_inputs=['*'], promotes_outputs=['*']
        )
        self.add_subsystem(
            'total_payload_mass', TotalPayload(), promotes_inputs=['*'], promotes_outputs=['*']
        )


class TotalPayload(om.ExplicitComponent):
    """Calculate the total payload mass."""

    def setup(self):
        add_aviary_input(self, Aircraft.CrewPayload.CARGO_MASS, units='lbm')
        add_aviary_input(self, Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS, units='lbm')

        add_aviary_output(self, Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(
            Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS,
            [Aircraft.CrewPayload.CARGO_MASS, Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS],
            val=1.0,
        )

    def compute(self, inputs, outputs):
        cargo_mass = inputs[Aircraft.CrewPayload.CARGO_MASS]
        passenger_payload_mass = inputs[Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS]

        outputs[Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS] = passenger_payload_mass + cargo_mass


class CargoMass(om.ExplicitComponent):
    """Calculate the total cargo mass."""

    def setup(self):
        add_aviary_input(self, Aircraft.CrewPayload.WING_CARGO, units='lbm')
        add_aviary_input(self, Aircraft.CrewPayload.MISC_CARGO, units='lbm')

        add_aviary_output(self, Aircraft.CrewPayload.CARGO_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(Aircraft.CrewPayload.CARGO_MASS, '*', val=1.0)

    def compute(self, inputs, outputs):
        wing_cargo = inputs[Aircraft.CrewPayload.WING_CARGO]
        misc_cargo = inputs[Aircraft.CrewPayload.MISC_CARGO]

        outputs[Aircraft.CrewPayload.CARGO_MASS] = wing_cargo + misc_cargo


class PaxPayloadMass(om.ExplicitComponent):
    """Calculate the mass of passengers and their baggage."""

    def initialize(self):
        add_aviary_option(self, Aircraft.CrewPayload.BAGGAGE_MASS_PER_PASSENGER, units='lbm')
        add_aviary_option(self, Aircraft.CrewPayload.MASS_PER_PASSENGER, units='lbm')
        add_aviary_option(self, Aircraft.CrewPayload.NUM_PASSENGERS)

    def setup(self):
        add_aviary_output(self, Aircraft.CrewPayload.PASSENGER_MASS_TOTAL, units='lbm')
        add_aviary_output(self, Aircraft.CrewPayload.BAGGAGE_MASS, units='lbm')
        add_aviary_output(self, Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS, units='lbm')

    def compute(self, inputs, outputs):
        passenger_count = self.options[Aircraft.CrewPayload.NUM_PASSENGERS]
        mass_per_passenger, _ = self.options[Aircraft.CrewPayload.MASS_PER_PASSENGER]
        baggage_mass_per_passenger, _ = self.options[
            Aircraft.CrewPayload.BAGGAGE_MASS_PER_PASSENGER
        ]

        pax_mass = mass_per_passenger * passenger_count
        baggage_mass = baggage_mass_per_passenger * passenger_count
        pax_payload = pax_mass + baggage_mass

        outputs[Aircraft.CrewPayload.PASSENGER_MASS_TOTAL] = pax_mass
        outputs[Aircraft.CrewPayload.BAGGAGE_MASS] = baggage_mass
        outputs[Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS] = pax_payload
