"""
OpenMDAO system for generating the aero tables that were typically printed in FLOPS.
"""
import numpy as np

import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input
from aviary.variable_info.variables import Aircraft, Mission


class DragPolar(om.ExplicitComponent):
    """
    This will need to create an instance of the dynamic aero group, perhaps as a
    subproblem, and run it at the table of mach numbers and lift coefficients. Right now,
    it is a placeholder, and also serves as a sink for all parts of the aircraft data
    structures that are passed to the dynamic portion, so that they can be overridden if
    needed.
    """

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        aviary_options = self.options['aviary_options']
        num_engine_type = len(aviary_options.get_val(Aircraft.Engine.NUM_ENGINES))

        add_aviary_input(self, Aircraft.Canard.WETTED_AREA, 0.0)
        add_aviary_input(self, Aircraft.Fuselage.WETTED_AREA, 0.0)
        add_aviary_input(self, Aircraft.HorizontalTail.WETTED_AREA, 0.0)
        add_aviary_input(self, Aircraft.Nacelle.WETTED_AREA, np.zeros(num_engine_type))
        add_aviary_input(self, Aircraft.VerticalTail.WETTED_AREA, 0.0)
        add_aviary_input(self, Aircraft.Wing.WETTED_AREA, 0.0)

        add_aviary_input(self, Aircraft.Canard.FINENESS, 0.0)
        add_aviary_input(self, Aircraft.Fuselage.FINENESS, 0.0)
        add_aviary_input(self, Aircraft.HorizontalTail.FINENESS, 0.0)
        add_aviary_input(self, Aircraft.Nacelle.FINENESS, np.zeros(num_engine_type))
        add_aviary_input(self, Aircraft.VerticalTail.FINENESS, 0.0)
        add_aviary_input(self, Aircraft.Wing.FINENESS, 0.0)

        add_aviary_input(self, Aircraft.Canard.CHARACTERISTIC_LENGTH, 0.0)
        add_aviary_input(self, Aircraft.Fuselage.CHARACTERISTIC_LENGTH, 0.0)
        add_aviary_input(self, Aircraft.HorizontalTail.CHARACTERISTIC_LENGTH, 0.0)
        add_aviary_input(self, Aircraft.Nacelle.CHARACTERISTIC_LENGTH,
                         np.zeros(num_engine_type))
        add_aviary_input(self, Aircraft.VerticalTail.CHARACTERISTIC_LENGTH, 0.0)
        add_aviary_input(self, Aircraft.Wing.CHARACTERISTIC_LENGTH, 0.0)

        add_aviary_input(self, Mission.Design.MACH, 0.0)
        add_aviary_input(self, Mission.Design.LIFT_COEFFICIENT, 0.0)
