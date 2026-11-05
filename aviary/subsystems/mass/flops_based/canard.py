import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission
from aviary.utils.utils import mass_to_force_english, mass_to_force_english_derivative


class CanardMass(om.ExplicitComponent):
    """
    Calculates the mass of the canard. The methodology is based on the FLOPS weight
    equations, modified to output mass instead of weight.
    """

    def initialize(self):
        add_aviary_option(self, Mission.GRAVITY, units='ft/s**2')

    def setup(self):
        add_aviary_input(self, Aircraft.Design.GROSS_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Canard.AREA, units='ft**2')
        add_aviary_input(self, Aircraft.Canard.TAPER_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.Canard.MASS_SCALER, units='unitless')

        add_aviary_output(self, Aircraft.Canard.MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        gravity = self.options[Mission.GRAVITY]

        togm = inputs[Aircraft.Design.GROSS_MASS]
        area = inputs[Aircraft.Canard.AREA]
        taper_ratio = inputs[Aircraft.Canard.TAPER_RATIO]
        scaler = inputs[Aircraft.Canard.MASS_SCALER]

        togw = mass_to_force_english((togm, 'lbm'), gravity)

        canard_weight = 0.53 * area * togw**0.2 * (taper_ratio + 0.5)
        outputs[Aircraft.Canard.MASS] = canard_weight * scaler

    def compute_partials(self, inputs, J):
        gravity = self.options[Mission.GRAVITY]

        togm = inputs[Aircraft.Design.GROSS_MASS]
        area = inputs[Aircraft.Canard.AREA]
        taper_ratio = inputs[Aircraft.Canard.TAPER_RATIO]
        scaler = inputs[Aircraft.Canard.MASS_SCALER]

        gross_weight = mass_to_force_english((togm, 'lbm'), gravity)
        dforce_dmass = mass_to_force_english_derivative(gravity)

        gross_weight_exp = gross_weight**0.2

        J[Aircraft.Canard.MASS, Aircraft.Canard.AREA] = (
            0.53 * scaler * (taper_ratio + 0.5) * gross_weight_exp
        )

        J[Aircraft.Canard.MASS, Aircraft.Canard.TAPER_RATIO] = (
            0.53 * area * scaler * gross_weight_exp
        )

        J[Aircraft.Canard.MASS, Aircraft.Canard.MASS_SCALER] = (
            0.53 * area * (taper_ratio + 0.5) * gross_weight_exp
        )

        J[Aircraft.Canard.MASS, Aircraft.Design.GROSS_MASS] = (
            dforce_dmass * (0.106 * area * scaler * (taper_ratio + 0.5)) / gross_weight**0.8
        )
