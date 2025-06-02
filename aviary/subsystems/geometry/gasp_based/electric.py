import numpy as np
import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft


class CableSize(om.ExplicitComponent):
    """Computation of cable length for hybrid electric augmented system."""

    def initialize(self):
        add_aviary_option(self, Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES)

    def setup(self):
        total_num_wing_engines = self.options[Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES]

        add_aviary_input(
            self,
            Aircraft.Engine.WING_LOCATIONS,
            shape=int(total_num_wing_engines / 2),
            units='unitless',
        )
        add_aviary_input(self, Aircraft.Wing.SPAN, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.AVG_DIAMETER, units='ft')

        add_aviary_output(self, Aircraft.Electrical.HYBRID_CABLE_LENGTH, units='ft')

        self.declare_partials(
            Aircraft.Electrical.HYBRID_CABLE_LENGTH,
            [
                Aircraft.Engine.WING_LOCATIONS,
                Aircraft.Wing.SPAN,
            ],
        )

        self.declare_partials(
            Aircraft.Electrical.HYBRID_CABLE_LENGTH, Aircraft.Fuselage.AVG_DIAMETER, val=2.0
        )

    def compute(self, inputs, outputs):
        eng_span_frac = np.sum(inputs[Aircraft.Engine.WING_LOCATIONS])
        wingspan = inputs[Aircraft.Wing.SPAN]
        cabin_width = inputs[Aircraft.Fuselage.AVG_DIAMETER]

        outputs[Aircraft.Electrical.HYBRID_CABLE_LENGTH] = (
            eng_span_frac * wingspan + 2.0 * cabin_width
        )

    def compute_partials(self, inputs, J):
        eng_span_frac = inputs[Aircraft.Engine.WING_LOCATIONS]
        wingspan = inputs[Aircraft.Wing.SPAN]

        J[
            Aircraft.Electrical.HYBRID_CABLE_LENGTH,
            Aircraft.Engine.WING_LOCATIONS,
        ] = wingspan
        J[Aircraft.Electrical.HYBRID_CABLE_LENGTH, Aircraft.Wing.SPAN] = np.sum(eng_span_frac)
