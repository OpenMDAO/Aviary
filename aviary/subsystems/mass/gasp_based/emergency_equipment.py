"""
Define utilities to calculate the estimated mass of emergency equipment mass
"""

import openmdao.api as om

from aviary.variable_info.functions import add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft

from aviary.constants import GRAV_ENGLISH_LBM


class EmergencyEquipment(om.ExplicitComponent):
    """
    Define the default component to calculate the estimated mass of emergency
    service equipment. The methodology is based on the
    GASP weight equations, modified to output mass instead of weight.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_PASSENGERS)

    def setup(self):
        add_aviary_output(self, Aircraft.Design.EMERGENCY_EQUIPMENT_MASS, units='lbm')

    def compute(self, inputs, outputs):
        num_pax = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]

        num_flight_attendants = 0
        if num_pax >= 20.0:
            num_flight_attendants = 1
        if num_pax >= 51.0:
            num_flight_attendants = 2
        if num_pax >= 101.0:
            num_flight_attendants = 3
        if num_pax >= 151.0:
            num_flight_attendants = 4
        if num_pax >= 201.0:
            num_flight_attendants = 5
        if num_pax >= 251.0:
            num_flight_attendants = 6

        emergency_wt = 0.0
        if num_pax > 5.0:
            emergency_wt = 10.0
        if num_pax > 9.0:
            emergency_wt = 15.0
        if num_pax >= 35.0:
            emergency_wt = 25.0 * num_flight_attendants + 15.0

        outputs[Aircraft.Design.EMERGENCY_EQUIPMENT_MASS] = emergency_wt / GRAV_ENGLISH_LBM
