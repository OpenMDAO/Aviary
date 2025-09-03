"""Contains any preliminary calculations on the fuselage."""

import numpy as np
import openmdao.api as om

from aviary.variable_info.enums import Verbosity
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Settings


class FuselagePrelim(om.ExplicitComponent):
    """
    Calculate fuselage average diameter and planform area defined by:
    Aircraft.Fuselage.AVG_DIAMETER = 0.5 * (max_height + max_width)
    Aircraft.Fuselage.PLANFORM_AREA = length * max_width.
    """

    def initialize(self):
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        add_aviary_input(self, Aircraft.Fuselage.LENGTH, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.MAX_HEIGHT, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.MAX_WIDTH, units='ft')

        add_aviary_output(self, Aircraft.Fuselage.AVG_DIAMETER, units='ft')
        add_aviary_output(self, Aircraft.Fuselage.PLANFORM_AREA, units='ft**2')

    def setup_partials(self):
        self.declare_partials(
            of=[Aircraft.Fuselage.AVG_DIAMETER],
            wrt=[Aircraft.Fuselage.MAX_HEIGHT, Aircraft.Fuselage.MAX_WIDTH],
            val=0.5,
        )
        self.declare_partials(
            of=[Aircraft.Fuselage.PLANFORM_AREA],
            wrt=[Aircraft.Fuselage.LENGTH, Aircraft.Fuselage.MAX_WIDTH],
        )

    def compute(self, inputs, outputs):
        verbosity = self.options[Settings.VERBOSITY]
        max_height = inputs[Aircraft.Fuselage.MAX_HEIGHT]
        max_width = inputs[Aircraft.Fuselage.MAX_WIDTH]
        length = inputs[Aircraft.Fuselage.LENGTH]
        if length <= 0.0:
            if verbosity > Verbosity.BRIEF:
                print('Aircraft.Fuselage.LENGTH must be positive.')

        avg_diameter = 0.5 * (max_height + max_width)
        outputs[Aircraft.Fuselage.AVG_DIAMETER] = avg_diameter

        outputs[Aircraft.Fuselage.PLANFORM_AREA] = length * max_width

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        max_width = inputs[Aircraft.Fuselage.MAX_WIDTH]
        length = inputs[Aircraft.Fuselage.LENGTH]

        partials[Aircraft.Fuselage.PLANFORM_AREA, Aircraft.Fuselage.LENGTH] = max_width

        partials[Aircraft.Fuselage.PLANFORM_AREA, Aircraft.Fuselage.MAX_WIDTH] = length


class SimpleCabinLayout(om.ExplicitComponent):
    """Given fuselage length, compute passenger compartment length."""

    def initialize(self):
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        add_aviary_input(self, Aircraft.Fuselage.LENGTH, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.MAX_HEIGHT, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.MAX_WIDTH, units='ft')

        add_aviary_output(self, Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH, units='ft')

    def setup_partials(self):
        self.declare_partials(
            of=[Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH],
            wrt=[Aircraft.Fuselage.LENGTH],
        )

    def compute(self, inputs, outputs):
        verbosity = self.options[Settings.VERBOSITY]

        length = inputs[Aircraft.Fuselage.LENGTH]
        max_height = inputs[Aircraft.Fuselage.MAX_HEIGHT]
        max_width = inputs[Aircraft.Fuselage.MAX_WIDTH]
        if length <= 0.0:
            if verbosity > Verbosity.BRIEF:
                print('Aircraft.Fuselage.LENGTH must be positive to use simple cabin layout.')
        if max_height <= 0.0 or max_width <= 0.0:
            if verbosity > Verbosity.BRIEF:
                print(
                    'Aircraft.Fuselage.MAX_HEIGHT & Aircraft.Fuselage.MAX_WIDTH must be positive.'
                )

        pax_compart_length = 0.6085 * length * (np.arctan(length / 59.0)) ** 1.1
        if pax_compart_length > 190.0:
            if verbosity > Verbosity.BRIEF:
                print(
                    'Passenger compartiment lenght is longer than recommended maximum length. '
                    'Suggest use detailed laylout algorithm.'
                )
        outputs[Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH] = pax_compart_length

    def compute_partials(self, inputs, partials):
        length = inputs[Aircraft.Fuselage.LENGTH]
        atan = np.arctan(length / 59.0)
        deriv = 0.6085 * (
            (atan) ** 1.1 + 1.1 * length * atan**0.1 / (1 + (length / 59.0) ** 2) / 59.0
        )
        partials[Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH, Aircraft.Fuselage.LENGTH] = deriv
