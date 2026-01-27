import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.utils.math import sigmoidX, dSigmoidXdx
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission


class OxygenSystemMass(om.ExplicitComponent):
    """
    Calculates mass of instrument group for transports and GA aircraft.
    The methodology is based on the GASP weight equations, modified to
    output mass instead of weight.

    ASSUMPTIONS: All engines have instrument mass that follows this equation
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_PASSENGERS)
        add_aviary_option(self, Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES)

    def setup(self):
        add_aviary_input(self, Mission.Design.GROSS_MASS, units='lbm')

        add_aviary_output(self, Aircraft.OxygenSystem.MASS, units='lbm')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        PAX = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]
        smooth = self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]

        gross_wt_initial = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM

        if PAX < 9:
            if smooth:
                oxygen_system_wt = 3 * sigmoidX(gross_wt_initial / 3000, 1.0, 0.01)
            else:
                if gross_wt_initial > 3000.0:  # note: this technically creates a discontinuity
                    oxygen_system_wt = 3.0
                else:
                    oxygen_system_wt = 0.0
        elif PAX >= 9 and PAX < 20:
            oxygen_system_wt = 10.0
        elif PAX >= 20 and PAX < 75:
            oxygen_system_wt = 20.0
        else:
            oxygen_system_wt = 50.0

        outputs[Aircraft.OxygenSystem.MASS] = oxygen_system_wt

    def compute_partials(self, inputs, J):
        PAX = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]
        smooth = self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]

        gross_wt_initial = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM

        if PAX < 9:
            if smooth:
                d_aux_wt_dgross_wt_initial = (
                    3 * dSigmoidXdx(gross_wt_initial / 3000, 1, 0.01) * 1 / 3000
                )
            else:
                if gross_wt_initial > 3000.0:
                    d_aux_wt_dgross_wt_initial = 0.0
        else:
            d_aux_wt_dgross_wt_initial = 0.0

        J[Aircraft.OxygenSystem.MASS, Mission.Design.GROSS_MASS] = d_aux_wt_dgross_wt_initial
