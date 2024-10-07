"""
Contains any preliminary calculations on the wing.
"""

import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft


class WingPrelim(om.ExplicitComponent):
    """
    preliminary calculations of wing aspect ratio
    """

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(self, Aircraft.Wing.AREA, val=0.0)
        add_aviary_input(self, Aircraft.Wing.GLOVE_AND_BAT, val=0.0)
        add_aviary_input(self, Aircraft.Wing.SPAN, val=0.0)

        add_aviary_output(self, Aircraft.Wing.ASPECT_RATIO, val=0.0)

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        area = inputs[Aircraft.Wing.AREA]
        glove_and_bat = inputs[Aircraft.Wing.GLOVE_AND_BAT]
        span = inputs[Aircraft.Wing.SPAN]

        AR = span ** 2 / (area - glove_and_bat)
        outputs[Aircraft.Wing.ASPECT_RATIO] = AR

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        area = inputs[Aircraft.Wing.AREA]
        glove_and_bat = inputs[Aircraft.Wing.GLOVE_AND_BAT]
        span = inputs[Aircraft.Wing.SPAN]

        denom = 1.0 / (area - glove_and_bat)

        partials[Aircraft.Wing.ASPECT_RATIO,
                 Aircraft.Wing.AREA] = -(span * denom) ** 2

        partials[Aircraft.Wing.ASPECT_RATIO,
                 Aircraft.Wing.GLOVE_AND_BAT] = (span * denom) ** 2

        partials[Aircraft.Wing.ASPECT_RATIO,
                 Aircraft.Wing.SPAN] = 2.0 * span * denom
