import numpy as np
import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft


class AvionicsMass(om.ExplicitComponent):
    """
    Calculates the mass of the avionics group using the transport/general aviation method.
    The methodology is based on the GASP weight equations, modified to output mass instead of weight.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_PASSENGERS)
        add_aviary_option(self, Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES)

    def setup(self):
        add_aviary_input(self, Aircraft.Design.GROSS_MASS, units='lbm')

        add_aviary_output(self, Aircraft.Avionics.MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        PAX = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]
        smooth = self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]

        gross_mass_initial = inputs[Aircraft.Design.GROSS_MASS]

        avionics_mass = 27.0

        # GASP avionics weight model was put together long before modern systems
        # came on-board, and should be updated.
        if PAX < 20:
            if smooth:
                # Exponential regression from four points:
                # (3000, 65), (5500, 113), (7500, 163), (11000, 340)
                # avionics_mass = 36.2 * exp(0.0002024 * gross_mass_initial)
                # Exponential regression from five points:
                # (0, 27), (3000, 65), (5500, 113), (7500, 163), (11000, 340)
                # avionics_mass = 30.03 * exp(0.0002262 * gross_mass_initial)
                # Should we use use 4 sigmoid functions (one for each transition zone) instead?
                avionics_mass = 35.538 * np.exp(0.0002 * gross_mass_initial)
            else:
                if gross_mass_initial >= 3000.0:
                    avionics_mass = 65.0
                if gross_mass_initial >= 5500.0:
                    avionics_mass = 113.0
                if gross_mass_initial >= 7500.0:
                    avionics_mass = 163.0
                if gross_mass_initial >= 11000.0:
                    avionics_mass = 340.0
        if PAX >= 20 and PAX < 30:
            avionics_mass = 400.0
        elif PAX >= 30 and PAX <= 50:
            avionics_mass = 500.0
        elif PAX > 50 and PAX <= 100:
            avionics_mass = 600.0
        if PAX > 100:
            avionics_mass = 2.8 * PAX + 1010.0

        outputs[Aircraft.Avionics.MASS] = avionics_mass

    def compute_partials(self, inputs, J):
        PAX = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]
        smooth = self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]

        gross_mass_initial = inputs[Aircraft.Design.GROSS_MASS]

        if PAX < 20:
            if smooth:
                davionics_mass_dgross_mass_initial = 0.0071076 * np.exp(0.0002 * gross_mass_initial)
            else:
                davionics_mass_dgross_mass_initial = 0.0
        else:
            davionics_mass_dgross_mass_initial = 0.0

        J[Aircraft.Avionics.MASS, Aircraft.Design.GROSS_MASS] = davionics_mass_dgross_mass_initial
