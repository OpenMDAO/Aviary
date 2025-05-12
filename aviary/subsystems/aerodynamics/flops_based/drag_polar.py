"""OpenMDAO system for generating the aero tables that were typically printed in FLOPS."""

import numpy as np
import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input, add_aviary_option
from aviary.variable_info.variables import Aircraft, Mission


class DragPolar(om.ExplicitComponent):
    """
    This will need to create an instance of the dynamic aero group, perhaps as a
    subproblem, and run it at the table of Mach numbers and lift coefficients. Right now,
    it is a placeholder, and also serves as a sink for all parts of the aircraft data
    structures that are passed to the dynamic portion, so that they can be overridden if
    needed.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Engine.NUM_ENGINES)

    def setup(self):
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])

        add_aviary_input(self, Aircraft.Canard.WETTED_AREA, units='ft**2')
        add_aviary_input(self, Aircraft.Fuselage.WETTED_AREA, units='ft**2')
        add_aviary_input(self, Aircraft.HorizontalTail.WETTED_AREA, units='ft**2')
        add_aviary_input(self, Aircraft.Nacelle.WETTED_AREA, shape=num_engine_type, units='ft**2')
        add_aviary_input(self, Aircraft.VerticalTail.WETTED_AREA, units='ft**2')
        add_aviary_input(self, Aircraft.Wing.WETTED_AREA, units='ft**2')

        add_aviary_input(self, Aircraft.Canard.FINENESS, units='unitless')
        add_aviary_input(self, Aircraft.Fuselage.FINENESS, units='unitless')
        add_aviary_input(self, Aircraft.HorizontalTail.FINENESS, units='unitless')
        add_aviary_input(self, Aircraft.Nacelle.FINENESS, shape=num_engine_type, units='unitless')
        add_aviary_input(self, Aircraft.VerticalTail.FINENESS, units='unitless')
        add_aviary_input(self, Aircraft.Wing.FINENESS, units='unitless')

        add_aviary_input(self, Aircraft.Canard.CHARACTERISTIC_LENGTH, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.CHARACTERISTIC_LENGTH, units='ft')
        add_aviary_input(self, Aircraft.HorizontalTail.CHARACTERISTIC_LENGTH, units='ft')
        add_aviary_input(
            self, Aircraft.Nacelle.CHARACTERISTIC_LENGTH, np.zeros(num_engine_type), units='ft'
        )
        add_aviary_input(self, Aircraft.VerticalTail.CHARACTERISTIC_LENGTH, units='ft')
        add_aviary_input(self, Aircraft.Wing.CHARACTERISTIC_LENGTH, units='ft')

        add_aviary_input(self, Mission.Design.MACH, units='unitless')
        add_aviary_input(self, Mission.Design.LIFT_COEFFICIENT, units='unitless')
