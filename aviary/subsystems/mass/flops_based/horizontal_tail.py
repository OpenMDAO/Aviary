import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission


class HorizontalTailMass(om.ExplicitComponent):
    '''
    Calculates the mass of the horizontal tail. The methodology is based on the FLOPS weight
    equations, modified to output mass instead of weight.
    '''

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(self, Aircraft.HorizontalTail.AREA, val=0.0)

        add_aviary_input(self, Aircraft.HorizontalTail.TAPER_RATIO, val=0.352)

        add_aviary_input(self, Mission.Design.GROSS_MASS, val=0.0)

        add_aviary_input(self, Aircraft.HorizontalTail.MASS_SCALER, val=1.0)

        add_aviary_output(self, Aircraft.HorizontalTail.MASS, val=0.0)

    def setup_partials(self):
        self.declare_partials("*", "*")

    def compute(self, inputs, outputs):
        area = inputs[Aircraft.HorizontalTail.AREA]
        gross_weight = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        scaler = inputs[Aircraft.HorizontalTail.MASS_SCALER]
        taper_ratio = inputs[Aircraft.HorizontalTail.TAPER_RATIO]

        outputs[Aircraft.HorizontalTail.MASS] = scaler * 0.530 * \
            area * gross_weight ** 0.20 * \
            (taper_ratio + 0.50) / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):
        area = inputs[Aircraft.HorizontalTail.AREA]
        gross_weight = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        scaler = inputs[Aircraft.HorizontalTail.MASS_SCALER]
        taper_ratio = inputs[Aircraft.HorizontalTail.TAPER_RATIO]

        gross_weight_exp = gross_weight ** 0.20

        J[Aircraft.HorizontalTail.MASS, Aircraft.HorizontalTail.AREA] = scaler * \
            0.530 * gross_weight_exp * (taper_ratio + 0.50) / GRAV_ENGLISH_LBM

        J[Aircraft.HorizontalTail.MASS, Aircraft.HorizontalTail.MASS_SCALER] = \
            0.530 * area * gross_weight_exp * \
            (taper_ratio + 0.50) / GRAV_ENGLISH_LBM

        J[Aircraft.HorizontalTail.MASS, Mission.Design.GROSS_MASS] = \
            scaler * 0.106 * area * gross_weight ** -0.8 * (taper_ratio + 0.50)

        J[Aircraft.HorizontalTail.MASS, Aircraft.HorizontalTail.TAPER_RATIO] = \
            scaler * 0.530 * area * gross_weight_exp / GRAV_ENGLISH_LBM


class AltHorizontalTailMass(om.ExplicitComponent):
    '''
    Calculates the mass of the horizontal tail using the alternate method.
    The methodology is based on the FLOPS weight equations, modified to
    output mass instead of weight.
    '''

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(self, Aircraft.HorizontalTail.AREA, val=0.0)

        add_aviary_input(self, Aircraft.HorizontalTail.MASS_SCALER, val=1.0)

        add_aviary_output(self, Aircraft.HorizontalTail.MASS, val=0.0)

    def setup_partials(self):
        self.declare_partials("*", "*")

    def compute(self, inputs, outputs):
        area = inputs[Aircraft.HorizontalTail.AREA]
        scaler = inputs[Aircraft.HorizontalTail.MASS_SCALER]

        outputs[Aircraft.HorizontalTail.MASS] = scaler * \
            5.4 * area / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):
        area = inputs[Aircraft.HorizontalTail.AREA]
        scaler = inputs[Aircraft.HorizontalTail.MASS_SCALER]

        J[Aircraft.HorizontalTail.MASS, Aircraft.HorizontalTail.AREA] = (
            5.4 * scaler / GRAV_ENGLISH_LBM
        )

        J[Aircraft.HorizontalTail.MASS, Aircraft.HorizontalTail.MASS_SCALER] = (
            5.4 * area / GRAV_ENGLISH_LBM
        )
