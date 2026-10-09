import openmdao.api as om

from aviary.utils.utils import mass_to_force_english, mass_to_force_english_derivative
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission


class FinMass(om.ExplicitComponent):
    """
    Calculates the mass of the fin(s). The methodology is based on the FLOPS weight
    equations, modified to output mass instead of weight.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Fins.NUM_FINS)
        add_aviary_option(self, Mission.GRAVITY, units='ft/s**2')

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
            gravity = self.options[Mission.GRAVITY]

            togm = inputs[Aircraft.Design.GROSS_MASS]
            area = inputs[Aircraft.Fins.AREA]
            taper_ratio = inputs[Aircraft.Fins.TAPER_RATIO]

            togw = mass_to_force_english((togm, 'lbm'), gravity)

            fin_weight = 0.32 * togw**0.3 * area**0.85 * (taper_ratio + 0.5) * num_fins
            outputs[Aircraft.Fins.MASS] = fin_weight * inputs[Aircraft.Fins.MASS_SCALER]

    def compute_partials(self, inputs, J):
        num_fins = self.options[Aircraft.Fins.NUM_FINS]
        if num_fins > 0:
            gravity = self.options[Mission.GRAVITY]

            togm = inputs[Aircraft.Design.GROSS_MASS]
            area = inputs[Aircraft.Fins.AREA]
            taper_ratio = inputs[Aircraft.Fins.TAPER_RATIO]
            scaler = inputs[Aircraft.Fins.MASS_SCALER]

            gross_weight = mass_to_force_english((togm, 'lbm'), gravity)
            dforce_dmass = mass_to_force_english_derivative(gravity)

            area_exp = area**0.85
            gross_weight_exp = gross_weight**0.3

            J[Aircraft.Fins.MASS, Aircraft.Fins.AREA] = (
                0.272 * num_fins * scaler * (taper_ratio + 0.5) * gross_weight_exp
            ) / area**0.15
            J[Aircraft.Fins.MASS, Aircraft.Fins.TAPER_RATIO] = (
                0.32 * area_exp * num_fins * scaler * gross_weight_exp
            )
            J[Aircraft.Fins.MASS, Aircraft.Fins.MASS_SCALER] = (
                0.32 * area_exp * num_fins * (taper_ratio + 0.5) * gross_weight_exp
            )
            J[Aircraft.Fins.MASS, Aircraft.Design.GROSS_MASS] = (
                dforce_dmass
                * (0.096 * area_exp * num_fins * scaler * (taper_ratio + 0.5))
                / gross_weight**0.7
            )
