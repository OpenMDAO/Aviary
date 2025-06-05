import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft


class TotalWettedArea(om.ExplicitComponent):
    """
    Sum of wetted areas of canard, fuselage, horizontal tail, nacelle, vertical tail and wing.
    It is simple enough to skip unit test.
    """

    def setup(self):
        add_aviary_input(self, Aircraft.Canard.WETTED_AREA, units='ft**2')
        add_aviary_input(self, Aircraft.Fuselage.WETTED_AREA, units='ft**2')
        add_aviary_input(self, Aircraft.HorizontalTail.WETTED_AREA, units='ft**2')
        add_aviary_input(self, Aircraft.Nacelle.TOTAL_WETTED_AREA, units='ft**2')
        add_aviary_input(self, Aircraft.VerticalTail.WETTED_AREA, units='ft**2')
        add_aviary_input(self, Aircraft.Wing.WETTED_AREA, units='ft**2')

        add_aviary_output(self, Aircraft.Design.TOTAL_WETTED_AREA, units='ft**2')

    def setup_partials(self):
        self.declare_partials('*', '*', val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs[Aircraft.Design.TOTAL_WETTED_AREA] = (
            inputs[Aircraft.Canard.WETTED_AREA]
            + inputs[Aircraft.Fuselage.WETTED_AREA]
            + inputs[Aircraft.HorizontalTail.WETTED_AREA]
            + inputs[Aircraft.Nacelle.TOTAL_WETTED_AREA]
            + inputs[Aircraft.VerticalTail.WETTED_AREA]
            + inputs[Aircraft.Wing.WETTED_AREA]
        )
