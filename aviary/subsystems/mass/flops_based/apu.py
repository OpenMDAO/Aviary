import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft


class TransportAPUMass(om.ExplicitComponent):
    """
    Calculates the mass of the auxiliary power unit. The methodology is based
    on the FLOPS weight equations, modified to output mass instead of weight.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_PASSENGERS)

    def setup(self):
        add_aviary_input(self, Aircraft.APU.MASS_SCALER, units='unitless')
        add_aviary_input(self, Aircraft.Fuselage.PLANFORM_AREA, units='ft**2')

        add_aviary_output(self, Aircraft.APU.MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        pax = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]
        scaler = inputs[Aircraft.APU.MASS_SCALER]
        planform = inputs[Aircraft.Fuselage.PLANFORM_AREA]

        outputs[Aircraft.APU.MASS] = (
            (54.0 * planform**0.3 + 5.4 * pax**0.9) * scaler / GRAV_ENGLISH_LBM
        )

    def compute_partials(self, inputs, J):
        pax = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]
        scaler = inputs[Aircraft.APU.MASS_SCALER]
        planform = inputs[Aircraft.Fuselage.PLANFORM_AREA]

        J[Aircraft.APU.MASS, Aircraft.APU.MASS_SCALER] = (
            54.0 * planform**0.3 + 5.4 * pax**0.9
        ) / GRAV_ENGLISH_LBM
        J[Aircraft.APU.MASS, Aircraft.Fuselage.PLANFORM_AREA] = (
            16.2 * planform**-0.7 * scaler / GRAV_ENGLISH_LBM
        )
