import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft


class TotalWettedArea(om.ExplicitComponent):

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(self, Aircraft.Canard.WETTED_AREA, 0.0)
        add_aviary_input(self, Aircraft.Fuselage.WETTED_AREA, 0.0)
        add_aviary_input(self, Aircraft.HorizontalTail.WETTED_AREA, 0.0)
        add_aviary_input(self, Aircraft.Nacelle.TOTAL_WETTED_AREA, 0.0)
        add_aviary_input(self, Aircraft.VerticalTail.WETTED_AREA, 0.0)
        add_aviary_input(self, Aircraft.Wing.WETTED_AREA, 0.0)

        add_aviary_output(self, Aircraft.Design.TOTAL_WETTED_AREA, 0.0)

    def setup_partials(self):
        self.declare_partials('*', '*', val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs[Aircraft.Design.TOTAL_WETTED_AREA] = (
            inputs[Aircraft.Canard.WETTED_AREA]
            + inputs[Aircraft.Fuselage.WETTED_AREA]
            + inputs[Aircraft.HorizontalTail.WETTED_AREA]
            + inputs[Aircraft.Nacelle.TOTAL_WETTED_AREA]
            + inputs[Aircraft.VerticalTail.WETTED_AREA]
            + inputs[Aircraft.Wing.WETTED_AREA])
