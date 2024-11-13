"""
Contains any preliminary calculations on the fuselage.
"""

import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft


class FuselagePrelim(om.ExplicitComponent):
    """
    Calculate fuselage average diameter and planform area defined by:
    Aircraft.Fuselage.AVG_DIAMETER = 0.5 * (max_height + max_width)
    Aircraft.Fuselage.PLANFORM_AREA = length * max_width
    """

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(self, Aircraft.Fuselage.LENGTH, val=0.0)
        add_aviary_input(self, Aircraft.Fuselage.MAX_HEIGHT, val=0.0)
        add_aviary_input(self, Aircraft.Fuselage.MAX_WIDTH, val=0.0)

        add_aviary_output(self, Aircraft.Fuselage.AVG_DIAMETER, val=0.0)
        add_aviary_output(self, Aircraft.Fuselage.PLANFORM_AREA, val=0.0)

    def setup_partials(self):
        self.declare_partials(of=[Aircraft.Fuselage.AVG_DIAMETER],
                              wrt=[Aircraft.Fuselage.MAX_HEIGHT,
                                   Aircraft.Fuselage.MAX_WIDTH],
                              val=0.5)
        self.declare_partials(of=[Aircraft.Fuselage.PLANFORM_AREA],
                              wrt=[Aircraft.Fuselage.LENGTH,
                                   Aircraft.Fuselage.MAX_WIDTH])

    def compute(self, inputs, outputs):
        max_height = inputs[Aircraft.Fuselage.MAX_HEIGHT]
        max_width = inputs[Aircraft.Fuselage.MAX_WIDTH]
        length = inputs[Aircraft.Fuselage.LENGTH]

        avg_diameter = 0.5 * (max_height + max_width)
        outputs[Aircraft.Fuselage.AVG_DIAMETER] = avg_diameter

        outputs[Aircraft.Fuselage.PLANFORM_AREA] = length * max_width

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        max_width = inputs[Aircraft.Fuselage.MAX_WIDTH]
        length = inputs[Aircraft.Fuselage.LENGTH]

        partials[Aircraft.Fuselage.PLANFORM_AREA,
                 Aircraft.Fuselage.LENGTH] = max_width

        partials[Aircraft.Fuselage.PLANFORM_AREA,
                 Aircraft.Fuselage.MAX_WIDTH] = length
