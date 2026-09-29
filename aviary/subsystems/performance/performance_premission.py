import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission


class PerformancePremission(om.ExplicitComponent):
    """Calculates the thrust-to-weight ratio and wing loading of the aircraft."""

    def initialize(self):
        add_aviary_option(self, Mission.GRAVITY, units='m/s**2')

    def setup(self):
        add_aviary_input(self, Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST, units='N')
        add_aviary_input(self, Aircraft.Design.GROSS_MASS, units='kg')
        add_aviary_input(self, Aircraft.Wing.AREA, units='m**2')

        add_aviary_output(self, Aircraft.Design.THRUST_TO_WEIGHT_RATIO, units='unitless')
        add_aviary_output(self, Aircraft.Design.WING_LOADING, units='N/m**2')

    def setup_partials(self):
        self.declare_partials(
            Aircraft.Design.THRUST_TO_WEIGHT_RATIO,
            [Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST, Aircraft.Design.GROSS_MASS],
        )
        self.declare_partials(
            Aircraft.Design.WING_LOADING, [Aircraft.Design.GROSS_MASS, Aircraft.Wing.AREA]
        )

    def compute(self, inputs, outputs):
        gravity = self.options[Mission.GRAVITY][0]

        thrust = inputs[Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST]
        weight = inputs[Aircraft.Design.GROSS_MASS] * gravity
        area = inputs[Aircraft.Wing.AREA]

        outputs[Aircraft.Design.THRUST_TO_WEIGHT_RATIO] = thrust / weight
        outputs[Aircraft.Design.WING_LOADING] = weight / area

    def compute_partials(self, inputs, J):
        gravity = self.options[Mission.GRAVITY][0]

        thrust = inputs[Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST]
        weight = inputs[Aircraft.Design.GROSS_MASS] * gravity
        area = inputs[Aircraft.Wing.AREA]

        J[Aircraft.Design.THRUST_TO_WEIGHT_RATIO, Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST] = (
            1 / (weight)
        )

        J[Aircraft.Design.THRUST_TO_WEIGHT_RATIO, Aircraft.Design.GROSS_MASS] = (
            -thrust * gravity / (weight**2)
        )

        J[Aircraft.Design.WING_LOADING, Aircraft.Design.GROSS_MASS] = gravity / area

        J[Aircraft.Design.WING_LOADING, Aircraft.Wing.AREA] = -weight / area**2
