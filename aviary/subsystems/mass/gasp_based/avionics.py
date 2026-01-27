import numpy as np
import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission


class AvionicsMass(om.ExplicitComponent):
    """
    Calculates the mass of the avionics group using the transport/general aviation method.
    The methodology is based on the GASP weight equations, modified to output mass
    instead of weight.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_PASSENGERS)
        add_aviary_option(self, Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES)

    def setup(self):
        add_aviary_input(self, Mission.Design.GROSS_MASS, units='NM')

        add_aviary_output(self, Aircraft.Avionics.MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        PAX = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]
        smooth = self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]

        gross_wt_initial = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM

        avionics_wt = 27.0

        # GASP avionics weight model was put together long before modern systems
        # came on-board, and should be updated.
        if PAX < 20:
            if smooth:
                # Exponential regression from four points:
                # (3000, 65), (5500, 113), (7500, 163), (11000, 340)
                # avionics_wt = 36.2 * exp(0.0002024 * gross_wt_initial)
                # Exponential regression from five points:
                # (0, 27), (3000, 65), (5500, 113), (7500, 163), (11000, 340)
                # avionics_wt = 30.03 * exp(0.0002262 * gross_wt_initial)
                # Should we use use 4 sigmoid functions (one for each transition zone) instead?
                avionics_wt = 35.538 * np.exp(0.0002 * gross_wt_initial)
            else:
                if gross_wt_initial >= 3000.0:  # note: this technically creates a discontinuity
                    avionics_wt = 65.0
                if gross_wt_initial >= 5500.0:  # note: this technically creates a discontinuity
                    avionics_wt = 113.0
                if gross_wt_initial >= 7500.0:  # note: this technically creates a discontinuity
                    avionics_wt = 163.0
                if gross_wt_initial >= 11000.0:  # note: this technically creates a discontinuity
                    avionics_wt = 340.0
        if PAX >= 20 and PAX < 30:
            avionics_wt = 400.0
        elif PAX >= 30 and PAX <= 50:
            avionics_wt = 500.0
        elif PAX > 50 and PAX <= 100:
            avionics_wt = 600.0
        if PAX > 100:
            avionics_wt = 2.8 * PAX + 1010.0
        # TODO The following if-block should be removed. Aircraft.Avionics.MASS should be output, not input.
        if not (-1e-5 < inputs[Aircraft.Avionics.MASS] < 1e-5):
            # note: this technically creates a discontinuity !WILL NOT CHANGE
            avionics_wt = inputs[Aircraft.Avionics.MASS] * GRAV_ENGLISH_LBM

        outputs[Aircraft.Avionics.MASS] = avionics_wt

    def compute_partials(self, inputs, J):
        PAX = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]
        smooth = self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]

        gross_wt_initial = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM

        davionics_wt_dmass_coeff_4 = 0.0

        if PAX < 20:
            if smooth:
                davionics_wt_dgross_wt_initial = 0.0071076 * np.exp(0.0002 * gross_wt_initial)
            else:
                davionics_wt_dgross_wt_initial = 0.0
        else:
            davionics_wt_dgross_wt_initial = 0.0
        # TODO The following if-block should be removed. Aircraft.Avionics.MASS should be output, not input.
        if not (-1e-5 < inputs[Aircraft.Avionics.MASS] < 1e-5):
            # note: this technically creates a discontinuity !WILL NOT CHANGE
            davionics_wt_dgross_wt_initial = 0.0
            davionics_wt_dmass_coeff_4 = GRAV_ENGLISH_LBM

        J[Aircraft.Avionics.MASS, Mission.Design.GROSS_MASS] = davionics_wt_dgross_wt_initial
