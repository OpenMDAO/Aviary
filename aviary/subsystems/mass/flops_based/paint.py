import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft


class PaintMass(om.ExplicitComponent):
    '''
    Calculates the mass of paint based on total wetted area.
    '''

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(self, Aircraft.Design.TOTAL_WETTED_AREA, val=0.0)

        add_aviary_input(self, Aircraft.Paint.MASS_PER_UNIT_AREA, val=0.0)

        add_aviary_output(self, Aircraft.Paint.MASS, val=0.0)

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(
        self, inputs, outputs, discrete_inputs=None, discrete_outputs=None
    ):
        wetted_area = inputs[Aircraft.Design.TOTAL_WETTED_AREA]
        mass_per_area = inputs[Aircraft.Paint.MASS_PER_UNIT_AREA]

        outputs[Aircraft.Paint.MASS] = wetted_area * mass_per_area

    def compute_partials(self, inputs, J):
        wetted_area = inputs[Aircraft.Design.TOTAL_WETTED_AREA]
        mass_per_area = inputs[Aircraft.Paint.MASS_PER_UNIT_AREA]

        J[Aircraft.Paint.MASS, Aircraft.Design.TOTAL_WETTED_AREA] = \
            mass_per_area

        J[Aircraft.Paint.MASS, Aircraft.Paint.MASS_PER_UNIT_AREA] = \
            wetted_area
