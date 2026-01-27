import numpy as np
import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM

from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission


class AntiIcingMass(om.ExplicitComponent):
    """
    Calculates the mass of the anti-icing system. The methodology is based
    on the GASP weight equations, modified to output mass instead of weight.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Engine.NUM_ENGINES)
        add_aviary_option(self, Aircraft.Propulsion.TOTAL_NUM_ENGINES)
        add_aviary_option(self, Aircraft.Engine.REFERENCE_SLS_THRUST, units='lbf')

    def setup(self):
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])

        add_aviary_input(self, Aircraft.AntiIcing.MASS_SCALER, units='unitless')
        add_aviary_input(self, Aircraft.Fuselage.MAX_WIDTH, units='ft')
        add_aviary_input(self, Aircraft.Nacelle.AVG_DIAMETER, shape=num_engine_type, units='ft')
        add_aviary_input(self, Aircraft.Wing.SPAN, units='ft')
        add_aviary_input(self, Aircraft.Wing.SWEEP, units='deg')
        add_aviary_input(
            self, Aircraft.Engine.SCALED_SLS_THRUST, shape=num_engine_type, units='lbf'
        )

        add_aviary_output(self, Aircraft.AntiIcing.MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        smooth = self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]

        wing_area = inputs[Aircraft.Wing.AREA]
        htail_area = inputs[Aircraft.HorizontalTail.AREA]
        vtail_area = inputs[Aircraft.VerticalTail.AREA]

        SSUM = wing_area + htail_area + vtail_area
        icing_wt = 22.7 * (SSUM**0.5) - 385.0

        if smooth:
            # This should be implemented:
            # icing_wt = smooth_max(icing_wt, 0.0, mu)
            pass
        else:
            if icing_wt < 0.0:  # note: this technically creates a discontinuity
                icing_wt = 0.0
        # TODO The following if-block should be removed. Aircraft.AntiIcing.MASS should be output, not input.
        if not (-1e-5 < inputs[Aircraft.AntiIcing.MASS] < 1e-5):
            # note: this technically creates a discontinuity !WILL NOT CHANGE
            icing_wt = inputs[Aircraft.AntiIcing.MASS] * GRAV_ENGLISH_LBM

        outputs[Aircraft.AntiIcing.MASS] = icing_wt

    def compute_partials(self, inputs, J):
        smooth = self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]

        wing_area = inputs[Aircraft.Wing.AREA]
        htail_area = inputs[Aircraft.HorizontalTail.AREA]
        vtail_area = inputs[Aircraft.VerticalTail.AREA]

        SSUM = wing_area + htail_area + vtail_area
        icing_wt = 22.7 * (SSUM**0.5) - 385.0

        dicing_weight_dwing_area = 0.5 * 22.7 * (SSUM**-0.5)
        dicing_weight_dhtail_area = 0.5 * 22.7 * (SSUM**-0.5)
        dicing_weight_dvtail_area = 0.5 * 22.7 * (SSUM**-0.5)
        dicing_weight_dmass_coeff_6 = 0.0

        if smooth:
            # The partials of smooth_max should be implemented here.
            pass
        else:
            if icing_wt < 0.0:  # note: this technically creates a discontinuity
                icing_wt = 0.0
                dicing_weight_dwing_area = 0.0
                dicing_weight_dhtail_area = 0.0
                dicing_weight_dvtail_area = 0.0
                dicing_weight_dmass_coeff_6 = 0.0
        # TODO The following if-block should be removed. Aircraft.AntiIcing.MASS should be output, not input.
        if not (-1e-5 < inputs[Aircraft.AntiIcing.MASS] < 1e-5):
            # note: this technically creates a discontinuity !WILL NOT CHANGE
            icing_wt = inputs[Aircraft.AntiIcing.MASS] * GRAV_ENGLISH_LBM
            dicing_weight_dwing_area = 0.0
            dicing_weight_dhtail_area = 0.0
            dicing_weight_dvtail_area = 0.0
            dicing_weight_dmass_coeff_6 = GRAV_ENGLISH_LBM

        J[Aircraft.AntiIcing.MASS, Mission.Design.GROSS_MASS] = dicing_weight_dmass_coeff_6
        J[Aircraft.AntiIcing.MASS, Aircraft.Wing.AREA] = dicing_weight_dwing_area
        J[Aircraft.AntiIcing.MASS, Aircraft.HorizontalTail.AREA] = dicing_weight_dhtail_area
        J[Aircraft.AntiIcing.MASS, Aircraft.VerticalTail.AREA] = dicing_weight_dvtail_area
