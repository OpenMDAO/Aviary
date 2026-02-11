"""
Define utilities to calculate the estimated mass of the crew (both flight and
non-flight) as well as their baggage.
"""

import openmdao.api as om

from aviary.variable_info.enums import GASPEngineType
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft
from aviary.constants import GRAV_ENGLISH_LBM


class NonFlightCrewMass(om.ExplicitComponent):
    """Calculate the estimated mass for the non-flight and their baggage."""

    def initialize(self):
        add_aviary_option(self, Aircraft.Engine.TYPE)
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_PASSENGERS)

    def setup(self):
        add_aviary_input(self, Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT)

        add_aviary_output(self, Aircraft.CrewPayload.NON_FLIGHT_CREW_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        PAX = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]
        water_mass_per_occupant = inputs[Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT]

        num_flight_attendants = 0
        if PAX >= 20.0:
            num_flight_attendants = 1
        if PAX >= 51.0:
            num_flight_attendants = 2
        if PAX >= 101.0:
            num_flight_attendants = 3
        if PAX >= 151.0:
            num_flight_attendants = 4
        if PAX >= 201.0:
            num_flight_attendants = 5
        if PAX >= 251.0:
            num_flight_attendants = 6

        # note: the average weight of a flight attendant was calculated using the following equation:
        # avg_wt = pct_male*avg_wt_male + pct_female*avg_wt_female where
        # pct_male = the percentage of US flight attendants that are male (based on data from
        # the women in aerospace international organization in 2020, which listed this percentage as
        # 20.8%)
        # avg_wt_male is the average weight of males according to the CDC, and is 199.8 lbf
        # pct_female is calculated from the same methods as pct_male, and results in 79.2%
        # avg_wt_female is the average weight of females according to the CDC, and is 170.8 lbf
        # the resulting value is that the average weight of the US flight attendant is 177 lbf
        flight_attendant_wt = 177 * num_flight_attendants

        if PAX >= 40.0:
            non_crew_bag_wt = 20.0 * num_flight_attendants
        else:
            non_crew_bag_wt = 10.0 * num_flight_attendants + 25.0

        water_mass = 0
        if PAX > 19.0:
            water_mass = water_mass_per_occupant * (num_flight_attendants)

        outputs[Aircraft.CrewPayload.NON_FLIGHT_CREW_MASS] = (
            flight_attendant_wt + non_crew_bag_wt
        ) / GRAV_ENGLISH_LBM + water_mass

    def compute_partials(self, inputs, J):
        PAX = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]
        num_flight_attendants = 0
        if PAX >= 20.0:
            num_flight_attendants = 1
        if PAX >= 51.0:
            num_flight_attendants = 2
        if PAX >= 101.0:
            num_flight_attendants = 3
        if PAX >= 151.0:
            num_flight_attendants = 4
        if PAX >= 201.0:
            num_flight_attendants = 5
        if PAX >= 251.0:
            num_flight_attendants = 6

        if PAX > 19.0:
            J[
                Aircraft.CrewPayload.NON_FLIGHT_CREW_MASS,
                Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT,
            ] = num_flight_attendants
        else:
            J[
                Aircraft.CrewPayload.NON_FLIGHT_CREW_MASS,
                Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT,
            ] = 0


class FlightCrewMass(om.ExplicitComponent):
    """Calculate the estimated mass for the flight crew and their baggage."""

    def initialize(self):
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_PASSENGERS)
        add_aviary_option(self, Aircraft.Engine.TYPE)

    def setup(self):
        add_aviary_input(self, Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT)

        add_aviary_output(self, Aircraft.CrewPayload.FLIGHT_CREW_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(
        self,
        inputs,
        outputs,
    ):
        PAX = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]
        engine_type = self.options[Aircraft.Engine.TYPE][0]

        water_mass_per_occupant = inputs[Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT]

        num_pilots = 1
        if PAX > 9.0:
            num_pilots = 2
        if engine_type is GASPEngineType.TURBOJET and PAX > 5.0:
            num_pilots = 2
        if PAX >= 351.0:
            num_pilots = 3

        # note: the average weight of a pilot was calculated using the following equation:
        # avg_wt = pct_male*avg_wt_male + pct_female*avg_wt_female where
        # pct_male = the percentage of US airline pilots that are male (based on data from
        # the center for aviation in 2018, which listed this percentage as 95.6%, and slightly
        # deflated to account for the upward trend in female pilots, resulting in an estimated
        # percentage of 94%)
        # avg_wt_male is the average weight of males according to the CDC, and is 199.8 lbf
        # pct_female is calculated from the same methods as pct_male, and results in 6%
        # avg_wt_female is the average weight of females according to the CDC, and is 170.8 lbf
        # the resulting value is that the average weight of the US airline pilot is 198 lbf
        pilot_wt = 198 * num_pilots

        if PAX >= 40.0:
            crew_bag_wt = 45 * num_pilots
        elif PAX < 20:
            crew_bag_wt = 25.0 * num_pilots
        else:
            crew_bag_wt = 10.0 * num_pilots + 25.0

        water_mass = 0.0
        if PAX > 19.0:
            water_mass = water_mass_per_occupant * num_pilots

        outputs[Aircraft.CrewPayload.FLIGHT_CREW_MASS] = (
            pilot_wt + crew_bag_wt
        ) / GRAV_ENGLISH_LBM + water_mass

    def compute_partials(self, inputs, J):
        PAX = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]
        engine_type = self.options[Aircraft.Engine.TYPE][0]
        num_pilots = 1
        if PAX > 9.0:
            num_pilots = 2
        if engine_type is GASPEngineType.TURBOJET and PAX > 5.0:
            num_pilots = 2
        if PAX >= 351.0:
            num_pilots = 3

        if PAX > 19.0:
            J[
                Aircraft.CrewPayload.FLIGHT_CREW_MASS,
                Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT,
            ] = num_pilots
        else:
            J[
                Aircraft.CrewPayload.FLIGHT_CREW_MASS,
                Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT,
            ] = 0
