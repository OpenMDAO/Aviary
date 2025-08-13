import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission


class PerformancePremission(om.ExplicitComponent):
    """Calculates the thrust-to-weight ratio and wing loading of the aircraft."""

    def setup(self):
        add_aviary_input(self, Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST, units='lbf')
        add_aviary_input(self, Mission.Design.GROSS_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Wing.AREA, units='ft**2')

        add_aviary_output(self, Aircraft.Design.THRUST_TO_WEIGHT_RATIO, units='unitless')
        add_aviary_output(self, Aircraft.Design.WING_LOADING, units='lbf/ft**2')

    def setup_partials(self):
        self.declare_partials(
            Aircraft.Design.THRUST_TO_WEIGHT_RATIO,
            [Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST, Mission.Design.GROSS_MASS],
        )
        self.declare_partials(
            Aircraft.Design.WING_LOADING, [Mission.Design.GROSS_MASS, Aircraft.Wing.AREA]
        )

    def compute(self, inputs, outputs):
        thrust = inputs[Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST]
        weight = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        area = inputs[Aircraft.Wing.AREA]

        outputs[Aircraft.Design.THRUST_TO_WEIGHT_RATIO] = thrust / weight
        outputs[Aircraft.Design.WING_LOADING] = weight / area

    def compute_partials(self, inputs, J):
        thrust = inputs[Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST]
        mass = inputs[Mission.Design.GROSS_MASS]
        area = inputs[Aircraft.Wing.AREA]

        J[Aircraft.Design.THRUST_TO_WEIGHT_RATIO, Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST] = (
            1 / (mass * GRAV_ENGLISH_LBM)
        )

        J[Aircraft.Design.THRUST_TO_WEIGHT_RATIO, Mission.Design.GROSS_MASS] = -thrust / (
            mass**2 * GRAV_ENGLISH_LBM
        )

        J[Aircraft.Design.WING_LOADING, Mission.Design.GROSS_MASS] = GRAV_ENGLISH_LBM / area

        J[Aircraft.Design.WING_LOADING, Aircraft.Wing.AREA] = -mass * GRAV_ENGLISH_LBM / area**2
