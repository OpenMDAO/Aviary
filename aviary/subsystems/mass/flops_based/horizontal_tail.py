import openmdao.api as om

from aviary.utils.utils import mass_to_force_english, mass_to_force_english_derivative
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission


class HorizontalTailMass(om.ExplicitComponent):
    """
    Calculates the mass of the horizontal tail. The methodology is based on the FLOPS weight
    equations, modified to output mass instead of weight.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.HorizontalTail.NUM_TAILS)
        add_aviary_option(self, Mission.GRAVITY, units='ft/s**2')

    def setup(self):
        add_aviary_input(self, Aircraft.HorizontalTail.AREA, units='ft**2')
        add_aviary_input(self, Aircraft.HorizontalTail.TAPER_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.Design.GROSS_MASS, units='lbm')
        add_aviary_input(self, Aircraft.HorizontalTail.MASS_SCALER, units='unitless')

        add_aviary_output(self, Aircraft.HorizontalTail.MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        num_tails = self.options[Aircraft.HorizontalTail.NUM_TAILS]
        gravity = self.options[Mission.GRAVITY]

        area = inputs[Aircraft.HorizontalTail.AREA]
        togm = inputs[Aircraft.Design.GROSS_MASS]
        scaler = inputs[Aircraft.HorizontalTail.MASS_SCALER]
        taper_ratio = inputs[Aircraft.HorizontalTail.TAPER_RATIO]

        gross_weight = mass_to_force_english((togm, 'lbm'), gravity)

        if num_tails == 1:
            outputs[Aircraft.HorizontalTail.MASS] = (
                scaler * 0.53 * area * gross_weight**0.20 * (taper_ratio + 0.50)
            )
        elif num_tails == 0:
            outputs[Aircraft.HorizontalTail.MASS] = 0.0
        else:
            raise UserWarning('FLOPS mass regressions do not support multiple horizontal tails.')

    def compute_partials(self, inputs, J):
        num_tails = self.options[Aircraft.HorizontalTail.NUM_TAILS]
        gravity = self.options[Mission.GRAVITY]

        area = inputs[Aircraft.HorizontalTail.AREA]
        togm = inputs[Aircraft.Design.GROSS_MASS]
        scaler = inputs[Aircraft.HorizontalTail.MASS_SCALER]
        taper_ratio = inputs[Aircraft.HorizontalTail.TAPER_RATIO]

        gross_weight = mass_to_force_english((togm, 'lbm'), gravity)
        dforce_dmass = mass_to_force_english_derivative(gravity)

        gross_weight_exp = gross_weight**0.20

        if num_tails == 1:
            J[Aircraft.HorizontalTail.MASS, Aircraft.HorizontalTail.AREA] = (
                scaler * 0.530 * gross_weight_exp * (taper_ratio + 0.50)
            )

            J[Aircraft.HorizontalTail.MASS, Aircraft.HorizontalTail.MASS_SCALER] = (
                0.530 * area * gross_weight_exp * (taper_ratio + 0.50)
            )

            J[Aircraft.HorizontalTail.MASS, Aircraft.Design.GROSS_MASS] = (
                dforce_dmass * scaler * 0.106 * area * gross_weight**-0.8 * (taper_ratio + 0.50)
            )

            J[Aircraft.HorizontalTail.MASS, Aircraft.HorizontalTail.TAPER_RATIO] = (
                scaler * 0.530 * area * gross_weight_exp
            )
        elif num_tails == 0:
            J[Aircraft.HorizontalTail.MASS, Aircraft.HorizontalTail.AREA] = 0.0
            J[Aircraft.HorizontalTail.MASS, Aircraft.HorizontalTail.MASS_SCALER] = 0.0
            J[Aircraft.HorizontalTail.MASS, Aircraft.Design.GROSS_MASS] = 0.0
            J[Aircraft.HorizontalTail.MASS, Aircraft.HorizontalTail.TAPER_RATIO] = 0.0


class AltHorizontalTailMass(om.ExplicitComponent):
    """
    Calculates the mass of the horizontal tail using the alternate method.
    The methodology is based on the FLOPS weight equations, modified to
    output mass instead of weight.
    """

    def setup(self):
        add_aviary_input(self, Aircraft.HorizontalTail.AREA, units='ft**2')
        add_aviary_input(self, Aircraft.HorizontalTail.MASS_SCALER, units='unitless')

        add_aviary_output(self, Aircraft.HorizontalTail.MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        area = inputs[Aircraft.HorizontalTail.AREA]
        scaler = inputs[Aircraft.HorizontalTail.MASS_SCALER]

        outputs[Aircraft.HorizontalTail.MASS] = scaler * 5.4 * area

    def compute_partials(self, inputs, J):
        area = inputs[Aircraft.HorizontalTail.AREA]
        scaler = inputs[Aircraft.HorizontalTail.MASS_SCALER]

        J[Aircraft.HorizontalTail.MASS, Aircraft.HorizontalTail.AREA] = 5.4 * scaler

        J[Aircraft.HorizontalTail.MASS, Aircraft.HorizontalTail.MASS_SCALER] = 5.4 * area
