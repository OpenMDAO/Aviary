import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.utils.math import d_smooth_max, smooth_max
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft


class AntiIcingMass(om.ExplicitComponent):
    """
    Calculates the mass of the anti-icing system. The methodology is based
    on the GASP weight equations, modified to output mass instead of weight.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES)
        self.options.declare('mu', default=10.0, types=float)

    def setup(self):
        add_aviary_input(self, Aircraft.Wing.AREA, units='ft**2')
        add_aviary_input(self, Aircraft.HorizontalTail.AREA, units='ft**2')
        add_aviary_input(self, Aircraft.VerticalTail.AREA, units='ft**2')

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
            mu = self.options['mu']
            icing_wt = smooth_max(icing_wt, 0.0, mu)
        else:
            if icing_wt < 0.0:
                icing_wt = 0.0

        outputs[Aircraft.AntiIcing.MASS] = icing_wt / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):
        smooth = self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]
        wing_area = inputs[Aircraft.Wing.AREA]
        htail_area = inputs[Aircraft.HorizontalTail.AREA]
        vtail_area = inputs[Aircraft.VerticalTail.AREA]

        SSUM = wing_area + htail_area + vtail_area
        icing_wt = 22.7 * (SSUM**0.5) - 385.0

        dicing_weight_dwing_area = 0.5 * 22.7 * (SSUM**-0.5) / GRAV_ENGLISH_LBM
        dicing_weight_dhtail_area = 0.5 * 22.7 * (SSUM**-0.5) / GRAV_ENGLISH_LBM
        dicing_weight_dvtail_area = 0.5 * 22.7 * (SSUM**-0.5) / GRAV_ENGLISH_LBM

        if smooth:
            import pdb

            # pdb.set_trace()
            mu = self.options['mu']
            sm_fac = d_smooth_max(icing_wt, 0.0, mu)
            dicing_weight_dwing_area = sm_fac * dicing_weight_dwing_area
            dicing_weight_dhtail_area = sm_fac * dicing_weight_dhtail_area
            dicing_weight_dvtail_area = sm_fac * dicing_weight_dvtail_area
        else:
            if icing_wt < 0.0:
                icing_wt = 0.0
                dicing_weight_dwing_area = 0.0
                dicing_weight_dhtail_area = 0.0
                dicing_weight_dvtail_area = 0.0

        J[Aircraft.AntiIcing.MASS, Aircraft.Wing.AREA] = dicing_weight_dwing_area
        J[Aircraft.AntiIcing.MASS, Aircraft.HorizontalTail.AREA] = dicing_weight_dhtail_area
        J[Aircraft.AntiIcing.MASS, Aircraft.VerticalTail.AREA] = dicing_weight_dvtail_area
