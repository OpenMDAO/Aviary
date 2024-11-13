import openmdao.api as om

from aviary.subsystems.geometry.flops_based.utils import (
    calc_lifting_surface_scaler, thickness_to_chord_scaler)
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft


class Canard(om.ExplicitComponent):
    """
    Calculate the wetted area of canard.
    """

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(self, Aircraft.Canard.AREA, 0.0)
        add_aviary_input(self, Aircraft.Canard.THICKNESS_TO_CHORD, 0.0)
        add_aviary_input(self, Aircraft.Canard.WETTED_AREA_SCALER, 1.0)

        add_aviary_output(self, Aircraft.Canard.WETTED_AREA, 0.0)

    def setup_partials(self):
        self.declare_partials(
            Aircraft.Canard.WETTED_AREA,
            [
                Aircraft.Canard.AREA, Aircraft.Canard.THICKNESS_TO_CHORD,
                Aircraft.Canard.WETTED_AREA_SCALER,
            ]
        )

    def compute(
        self, inputs, outputs, discrete_inputs=None, discrete_outputs=None
    ):
        area = inputs[Aircraft.Canard.AREA]

        if area <= 0.0:
            outputs[Aircraft.Canard.WETTED_AREA] = 0.0

            return

        thickness_to_chord = inputs[Aircraft.Canard.THICKNESS_TO_CHORD]
        XMULTC = calc_lifting_surface_scaler(thickness_to_chord)
        scaler = inputs[Aircraft.Canard.WETTED_AREA_SCALER]

        wetted_area = scaler * XMULTC * area

        outputs[Aircraft.Canard.WETTED_AREA] = wetted_area

    def compute_partials(self, inputs, J, discrete_inputs=None):
        area = inputs[Aircraft.Canard.AREA]

        thickness_to_chord = inputs[Aircraft.Canard.THICKNESS_TO_CHORD]
        XMULTC = calc_lifting_surface_scaler(thickness_to_chord)
        scaler = inputs[Aircraft.Canard.WETTED_AREA_SCALER]

        J[
            Aircraft.Canard.WETTED_AREA,
            Aircraft.Canard.AREA
        ] = scaler * XMULTC

        J[
            Aircraft.Canard.WETTED_AREA,
            Aircraft.Canard.THICKNESS_TO_CHORD
        ] = scaler * thickness_to_chord_scaler * area

        J[
            Aircraft.Canard.WETTED_AREA,
            Aircraft.Canard.WETTED_AREA_SCALER
        ] = XMULTC * area
