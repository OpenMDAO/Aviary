import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission


class FinMass(om.ExplicitComponent):
    """
    Calculates the mass of the fin(s). The methodology is based on the FLOPS weight
    equations, modified to output mass instead of weight.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Fins.NUM_FINS)

    def setup(self):
        add_aviary_input(self, Aircraft.Design.GROSS_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Fins.AREA, units='ft**2')
        add_aviary_input(self, Aircraft.Fins.TAPER_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.Fins.MASS_SCALER, units='unitless')

        add_aviary_output(self, Aircraft.Fins.MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        num_fins = self.options[Aircraft.Fins.NUM_FINS]
        if num_fins > 0:
            togw = inputs[Aircraft.Design.GROSS_MASS]
            area = inputs[Aircraft.Fins.AREA]
            taper_ratio = inputs[Aircraft.Fins.TAPER_RATIO]

            fin_weight = 0.32 * togw**0.3 * area**0.85 * (taper_ratio + 0.5) * num_fins
            outputs[Aircraft.Fins.MASS] = fin_weight * inputs[Aircraft.Fins.MASS_SCALER]

    def compute_partials(self, inputs, J):
        num_fins = self.options[Aircraft.Fins.NUM_FINS]
        if num_fins > 0:
            area = inputs[Aircraft.Fins.AREA]
            taper_ratio = inputs[Aircraft.Fins.TAPER_RATIO]
            scaler = inputs[Aircraft.Fins.MASS_SCALER]
            gross_mass = inputs[Aircraft.Design.GROSS_MASS]

            area_exp = area**0.85
            gross_mass_exp = gross_mass**0.3

            J[Aircraft.Fins.MASS, Aircraft.Fins.AREA] = (
                0.272 * num_fins * scaler * (taper_ratio + 0.5) * gross_mass_exp
            ) / area**0.15
            J[Aircraft.Fins.MASS, Aircraft.Fins.TAPER_RATIO] = (
                0.32 * area_exp * num_fins * scaler * gross_mass_exp
            )
            J[Aircraft.Fins.MASS, Aircraft.Fins.MASS_SCALER] = (
                0.32 * area_exp * num_fins * (taper_ratio + 0.5) * gross_mass_exp
            )
            J[Aircraft.Fins.MASS, Aircraft.Design.GROSS_MASS] = (
                0.096 * area_exp * num_fins * scaler * (taper_ratio + 0.5)
            ) / gross_mass**0.7
