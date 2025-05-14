import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission


class SurfaceControlMass(om.ExplicitComponent):
    """
    Calculate the mass of the surface controls. The methodology is based on the
    FLOPS weight equations, modified to output mass instead of weight.
    """

    def initialize(self):
        add_aviary_option(self, Mission.Constraints.MAX_MACH)

    def setup(self):
        add_aviary_input(self, Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, units='unitless')
        add_aviary_input(self, Mission.Design.GROSS_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Wing.CONTROL_SURFACE_AREA_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.Wing.AREA, units='ft**2')

        add_aviary_output(self, Aircraft.Wing.SURFACE_CONTROL_MASS, units='lbm')
        add_aviary_output(self, Aircraft.Wing.CONTROL_SURFACE_AREA, units='ft**2')

        self.declare_partials(Aircraft.Wing.SURFACE_CONTROL_MASS, '*')
        self.declare_partials(
            Aircraft.Wing.CONTROL_SURFACE_AREA,
            [Aircraft.Wing.CONTROL_SURFACE_AREA_RATIO, Aircraft.Wing.AREA],
        )

    def compute(self, inputs, outputs):
        scaler = inputs[Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER]
        max_mach = self.options[Mission.Constraints.MAX_MACH]
        gross_weight = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        flap_ratio = inputs[Aircraft.Wing.CONTROL_SURFACE_AREA_RATIO]
        wing_area = inputs[Aircraft.Wing.AREA]

        surface_flap_area = flap_ratio * wing_area

        surface_ctrls_wt = (
            1.1 * max_mach**0.52 * surface_flap_area**0.6 * gross_weight**0.32 * scaler
        )

        outputs[Aircraft.Wing.CONTROL_SURFACE_AREA] = surface_flap_area

        outputs[Aircraft.Wing.SURFACE_CONTROL_MASS] = surface_ctrls_wt / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):
        scaler = inputs[Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER]
        max_mach = self.options[Mission.Constraints.MAX_MACH]
        gross_weight = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        flap_ratio = inputs[Aircraft.Wing.CONTROL_SURFACE_AREA_RATIO]
        wing_area = inputs[Aircraft.Wing.AREA]

        surface_flap_area = flap_ratio * wing_area

        max_mach_exp = max_mach**0.52
        surface_area_exp = surface_flap_area**0.6
        gross_weight_exp = gross_weight**0.32

        J[Aircraft.Wing.SURFACE_CONTROL_MASS, Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER] = (
            1.1 * max_mach_exp * surface_area_exp * gross_weight_exp / GRAV_ENGLISH_LBM
        )

        J[Aircraft.Wing.SURFACE_CONTROL_MASS, Mission.Design.GROSS_MASS] = (
            1.1 * max_mach_exp * surface_area_exp * 0.32 * gross_weight ** (0.32 - 1) * scaler
        )

        J[Aircraft.Wing.SURFACE_CONTROL_MASS, Aircraft.Wing.CONTROL_SURFACE_AREA_RATIO] = (
            1.1
            * max_mach_exp
            * 0.6
            * surface_flap_area ** (0.6 - 1)
            * gross_weight_exp
            * wing_area
            * scaler
            / GRAV_ENGLISH_LBM
        )

        J[Aircraft.Wing.SURFACE_CONTROL_MASS, Aircraft.Wing.AREA] = (
            1.1
            * max_mach_exp
            * 0.6
            * surface_flap_area ** (0.6 - 1)
            * gross_weight_exp
            * flap_ratio
            * scaler
            / GRAV_ENGLISH_LBM
        )

        J[Aircraft.Wing.CONTROL_SURFACE_AREA, Aircraft.Wing.CONTROL_SURFACE_AREA_RATIO] = wing_area
        J[Aircraft.Wing.CONTROL_SURFACE_AREA, Aircraft.Wing.AREA] = flap_ratio


class AltSurfaceControlMass(om.ExplicitComponent):
    """
    Calculate the mass of the surface controls using the alternate method.
    The methodology is based on the FLOPS weight equations, modified to
    output mass instead of weight.
    """

    def setup(self):
        add_aviary_input(self, Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, units='unitless')
        add_aviary_input(self, Aircraft.Wing.AREA, units='ft**2')
        add_aviary_input(self, Aircraft.Wing.CONTROL_SURFACE_AREA_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.HorizontalTail.WETTED_AREA, units='ft**2')
        add_aviary_input(self, Aircraft.HorizontalTail.THICKNESS_TO_CHORD, units='unitless')
        add_aviary_input(self, Aircraft.VerticalTail.AREA, units='ft**2')

        add_aviary_output(self, Aircraft.Wing.CONTROL_SURFACE_AREA, units='ft**2')
        add_aviary_output(self, Aircraft.Wing.SURFACE_CONTROL_MASS, units='lbm')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        scaler = inputs[Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER]
        wing_area = inputs[Aircraft.Wing.AREA]
        flap_ratio = inputs[Aircraft.Wing.CONTROL_SURFACE_AREA_RATIO]
        htail_area = inputs[Aircraft.HorizontalTail.WETTED_AREA]
        htail_TCR = inputs[Aircraft.HorizontalTail.THICKNESS_TO_CHORD]
        vtail_area = inputs[Aircraft.VerticalTail.AREA]

        surface_ctrls_wt = (
            480 + 0.99 * wing_area + 2.5 * htail_area / (2 + 0.387 * htail_TCR) + 1.6 * vtail_area
        ) * scaler

        outputs[Aircraft.Wing.SURFACE_CONTROL_MASS] = surface_ctrls_wt / GRAV_ENGLISH_LBM

        outputs[Aircraft.Wing.CONTROL_SURFACE_AREA] = flap_ratio * wing_area

    def compute_partials(self, inputs, J):
        scaler = inputs[Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER]
        wing_area = inputs[Aircraft.Wing.AREA]
        flap_ratio = inputs[Aircraft.Wing.CONTROL_SURFACE_AREA_RATIO]
        htail_area = inputs[Aircraft.HorizontalTail.WETTED_AREA]
        htail_TCR = inputs[Aircraft.HorizontalTail.THICKNESS_TO_CHORD]
        vtail_area = inputs[Aircraft.VerticalTail.AREA]

        J[Aircraft.Wing.SURFACE_CONTROL_MASS, Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER] = (
            480 + 0.99 * wing_area + 2.5 * htail_area / (2 + 0.387 * htail_TCR) + 1.6 * vtail_area
        ) / GRAV_ENGLISH_LBM

        J[Aircraft.Wing.SURFACE_CONTROL_MASS, Aircraft.Wing.AREA] = 0.99 * scaler / GRAV_ENGLISH_LBM
        J[Aircraft.Wing.SURFACE_CONTROL_MASS, Aircraft.HorizontalTail.WETTED_AREA] = (
            2.5 / (2 + 0.387 * htail_TCR) * scaler / GRAV_ENGLISH_LBM
        )

        J[Aircraft.Wing.SURFACE_CONTROL_MASS, Aircraft.HorizontalTail.THICKNESS_TO_CHORD] = (
            -2.5 * htail_area / (2 + 0.387 * htail_TCR) ** 2 * 0.387 * scaler / GRAV_ENGLISH_LBM
        )

        J[Aircraft.Wing.SURFACE_CONTROL_MASS, Aircraft.VerticalTail.AREA] = (
            1.6 * scaler / GRAV_ENGLISH_LBM
        )

        J[Aircraft.Wing.CONTROL_SURFACE_AREA, Aircraft.Wing.CONTROL_SURFACE_AREA_RATIO] = wing_area

        J[Aircraft.Wing.CONTROL_SURFACE_AREA, Aircraft.Wing.AREA] = flap_ratio
