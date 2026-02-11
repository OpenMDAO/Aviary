import openmdao.api as om

from aviary.variable_info.functions import add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft

from aviary.constants import GRAV_ENGLISH_LBM


class APUMass(om.ExplicitComponent):
    """
    Calculates the mass of the auxiliary power unit. The methodology is based
    on the GASP weight equations, modified to output mass instead of weight.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_PASSENGERS)

    def setup(self):
        add_aviary_output(self, Aircraft.APU.MASS, units='lbm')

    def compute(self, inputs, outputs):
        PAX = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]

        if PAX > 35.0:
            outputs[Aircraft.APU.MASS] = (26.2 * PAX**0.944 - 13.6 * PAX) / GRAV_ENGLISH_LBM
        else:
            outputs[Aircraft.APU.MASS] = 0.0
