"""
Define utilities to calculate the estimated mass of passenger service
equipment.
"""

import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft


class PassengerServiceMass(om.ExplicitComponent):
    """
    Define the default component to calculate the estimated mass of passenger
    service equipment. The methodology is based on the
    GASP weight equations, modified to output mass instead of weight.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_PASSENGERS)

    def setup(self):
        add_aviary_input(
            self, Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_PER_PASSENGER, units='unitless'
        )
        add_aviary_input(self, Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT, units='unitless')
        add_aviary_input(self, Aircraft.CrewPayload.CATERING_ITEMS_MASS_PER_PASSENGER, units='lbm')

        add_aviary_output(self, Aircraft.CrewPayload.PASSENGER_SERVICE_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        num_pax = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]
        service_mass_per_passenger = inputs[
            Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_PER_PASSENGER
        ]
        water_mass_per_occupant = inputs[Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT]
        catering_mass_per_passenger = inputs[Aircraft.CrewPayload.CATERING_ITEMS_MASS_PER_PASSENGER]

        num_lavatories = 0
        if num_pax > 25.0:
            num_lavatories = 1
        if num_pax >= 51.0:
            num_lavatories = 2
        if num_pax >= 101.0:
            num_lavatories = 3
        if num_pax >= 151.0:
            num_lavatories = 4
        if num_pax >= 201.0:
            num_lavatories = 5
        if num_pax >= 251.0:
            num_lavatories = 6

        service_wt = 0.0
        if num_pax > 9.0:
            service_wt = (
                service_mass_per_passenger * num_pax * GRAV_ENGLISH_LBM + 16.0 * num_lavatories
            )

        water_wt = 0.0
        if num_pax > 19.0:
            water_wt = water_mass_per_occupant * num_pax * GRAV_ENGLISH_LBM

        catering_wt = 0.0
        if num_pax > 19.0:
            catering_wt = catering_mass_per_passenger * num_pax * GRAV_ENGLISH_LBM

        outputs[Aircraft.CrewPayload.PASSENGER_SERVICE_MASS] = service_wt + water_wt + catering_wt

    def compute_partials(self, J):
        num_pax = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]

        if num_pax > 19.0:
            J[
                Aircraft.CrewPayload.PASSENGER_SERVICE_MASS,
                Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_PER_PASSENGER,
            ] = 1

            J[
                Aircraft.CrewPayload.PASSENGER_SERVICE_MASS,
                Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT,
            ] = 1

            J[
                Aircraft.CrewPayload.PASSENGER_SERVICE_MASS,
                Aircraft.CrewPayload.CATERING_ITEMS_MASS_PER_PASSENGER,
            ] = 1

        else:
            J[
                Aircraft.CrewPayload.PASSENGER_SERVICE_MASS,
                Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_PER_PASSENGER,
            ] = 0

            J[
                Aircraft.CrewPayload.PASSENGER_SERVICE_MASS,
                Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT,
            ] = 0

            J[
                Aircraft.CrewPayload.PASSENGER_SERVICE_MASS,
                Aircraft.CrewPayload.CATERING_ITEMS_MASS_PER_PASSENGER,
            ] = 0
