import numpy as np
import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft


class NacelleTotalWettedArea(om.ExplicitComponent):
    """
    Sums the wetted area of all Nacelles.

    This is separated so that individual nacelle areas can be overrriden without breaking the
    summation. Note that this contributes to the total wetted area, which is used by the mass
    subsystem to calculate the weight of the paint.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Engine.NUM_ENGINES)

    def setup(self):
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])
        add_aviary_input(self, Aircraft.Nacelle.WETTED_AREA, shape=num_engine_type, units='ft**2')
        add_aviary_output(self, Aircraft.Nacelle.TOTAL_WETTED_AREA, units='ft**2')

    def setup_partials(self):
        num_engines = self.options[Aircraft.Engine.NUM_ENGINES]

        self.declare_partials(
            Aircraft.Nacelle.TOTAL_WETTED_AREA,
            Aircraft.Nacelle.WETTED_AREA,
            val=num_engines,
        )

    def compute(self, inputs, outputs):
        num_engines = self.options[Aircraft.Engine.NUM_ENGINES]
        wetted_area = inputs[Aircraft.Nacelle.WETTED_AREA]

        total_wetted_area = sum(num_engines * wetted_area)
        outputs[Aircraft.Nacelle.TOTAL_WETTED_AREA] = total_wetted_area
