import numpy as np
import openmdao.api as om

from aviary.subsystems.geometry.flops_based.utils import Names
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft


class WingCharacteristicLength(om.ExplicitComponent):
    """
    Calculate the characteristic length and fineness ratio of the wing.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION)

    def setup(self):
        self.add_input(Names.CROOT, 0.0, units='unitless')

        add_aviary_input(self, Aircraft.Wing.AREA, units='ft**2')
        add_aviary_input(self, Aircraft.Wing.ASPECT_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.Wing.GLOVE_AND_BAT, units='ft**2')
        add_aviary_input(self, Aircraft.Wing.TAPER_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD, units='unitless')

        add_aviary_output(self, Aircraft.Wing.CHARACTERISTIC_LENGTH, units='ft')
        add_aviary_output(self, Aircraft.Wing.FINENESS, units='unitless')

    def setup_partials(self):
        wrt = [
            Aircraft.Wing.AREA,
            Aircraft.Wing.ASPECT_RATIO,
            Aircraft.Wing.GLOVE_AND_BAT,
        ]

        if self.options[Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION]:
            wrt = [
                Names.CROOT,
                Aircraft.Wing.TAPER_RATIO,
            ]

        self.declare_partials(Aircraft.Wing.CHARACTERISTIC_LENGTH, wrt)

        self.declare_partials(Aircraft.Wing.FINENESS, Aircraft.Wing.THICKNESS_TO_CHORD, val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        area = inputs[Aircraft.Wing.AREA]
        glove_and_bat = inputs[Aircraft.Wing.GLOVE_AND_BAT]
        aspect_ratio = inputs[Aircraft.Wing.ASPECT_RATIO]

        if self.options[Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION]:
            taper_ratio = inputs[Aircraft.Wing.TAPER_RATIO]
            CROOT = inputs[Names.CROOT]

            length = (
                2.0 * CROOT * (1.0 + taper_ratio + taper_ratio**2.0) / (3.0 + 3.0 * taper_ratio)
            )
        else:
            length = ((area - glove_and_bat) / aspect_ratio) ** 0.5

        outputs[Aircraft.Wing.CHARACTERISTIC_LENGTH] = length

        thickness_to_chord = inputs[Aircraft.Wing.THICKNESS_TO_CHORD]

        outputs[Aircraft.Wing.FINENESS] = thickness_to_chord

    def compute_partials(self, inputs, J, discrete_inputs=None):
        if self.options[Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION]:
            taper_ratio = inputs[Aircraft.Wing.TAPER_RATIO]
            CROOT = inputs[Names.CROOT]

            a = 2.0 * (1.0 + taper_ratio + taper_ratio**2.0)
            f = a * CROOT
            df = 2.0 * CROOT * (1.0 + 2.0 * taper_ratio)
            g = 3.0 + 3.0 * taper_ratio
            dg = 3.0

            J[Aircraft.Wing.CHARACTERISTIC_LENGTH, Names.CROOT] = a / g

            J[Aircraft.Wing.CHARACTERISTIC_LENGTH, Aircraft.Wing.TAPER_RATIO] = (
                df * g - f * dg
            ) / g**2

        else:
            area = inputs[Aircraft.Wing.AREA]
            glove_and_bat = inputs[Aircraft.Wing.GLOVE_AND_BAT]
            aspect_ratio = inputs[Aircraft.Wing.ASPECT_RATIO]

            a = area - glove_and_bat
            f = 0.5 * (a / aspect_ratio) ** -0.5
            df = f / aspect_ratio

            J[Aircraft.Wing.CHARACTERISTIC_LENGTH, Aircraft.Wing.AREA] = df

            J[Aircraft.Wing.CHARACTERISTIC_LENGTH, Aircraft.Wing.GLOVE_AND_BAT] = -df

            J[Aircraft.Wing.CHARACTERISTIC_LENGTH, Aircraft.Wing.ASPECT_RATIO] = (
                -f * a / aspect_ratio**2.0
            )


class BWBWingCharacteristicLength(om.ExplicitComponent):
    """
    Calculate the characteristic length and fineness ratio of the wing of BWB.
    """

    def setup(self):
        add_aviary_input(self, Aircraft.Wing.SPAN, units='ft')
        add_aviary_input(self, Aircraft.Wing.AREA, units='ft**2')
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD, units='unitless')

        add_aviary_output(self, Aircraft.Wing.CHARACTERISTIC_LENGTH, units='ft')
        add_aviary_output(self, Aircraft.Wing.FINENESS, units='unitless')

    def setup_partials(self):
        wrt = [
            Aircraft.Wing.SPAN,
            Aircraft.Wing.AREA,
        ]

        self.declare_partials(Aircraft.Wing.CHARACTERISTIC_LENGTH, wrt)

        self.declare_partials(Aircraft.Wing.FINENESS, Aircraft.Wing.THICKNESS_TO_CHORD, val=1.0)

    def compute(self, inputs, outputs):
        wing_span = inputs[Aircraft.Wing.SPAN]
        wing_area = inputs[Aircraft.Wing.AREA]
        cl = wing_area / wing_span
        outputs[Aircraft.Wing.CHARACTERISTIC_LENGTH] = cl

        # a little redundant but FLOPS used two variables
        thickness_to_chord = inputs[Aircraft.Wing.THICKNESS_TO_CHORD]
        outputs[Aircraft.Wing.FINENESS] = thickness_to_chord

    def compute_partials(self, inputs, J):
        wing_span = inputs[Aircraft.Wing.SPAN]
        wing_area = inputs[Aircraft.Wing.AREA]
        J[Aircraft.Wing.CHARACTERISTIC_LENGTH, Aircraft.Wing.SPAN] = -wing_area / wing_span**2
        J[Aircraft.Wing.CHARACTERISTIC_LENGTH, Aircraft.Wing.AREA] = 1.0 / wing_span


class OtherCharacteristicLengths(om.ExplicitComponent):
    """
    Calculate the characteristic length and fineness ratio of the
    canard, fuselage, horizontal tail, nacelle, and vertical tail.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Engine.NUM_ENGINES)

    def setup(self):
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])

        self.add_input(Names.CROOT, 0.0, units='unitless')

        add_aviary_input(self, Aircraft.Canard.AREA, units='ft**2')
        add_aviary_input(self, Aircraft.Canard.ASPECT_RATIO, units='unitless')
        # add_aviary_input(self, Aircraft.Canard.LAMINAR_FLOW_LOWER, 0.0)
        # add_aviary_input(self, Aircraft.Canard.LAMINAR_FLOW_UPPER, 0.0)
        add_aviary_input(self, Aircraft.Canard.THICKNESS_TO_CHORD, units='unitless')

        add_aviary_input(self, Aircraft.Fuselage.AVG_DIAMETER, units='ft')
        # add_aviary_input(self, Aircraft.Fuselage.LAMINAR_FLOW_LOWER, 0.0)
        # add_aviary_input(self, Aircraft.Fuselage.LAMINAR_FLOW_UPPER, 0.0)
        add_aviary_input(self, Aircraft.Fuselage.LENGTH, units='ft')

        add_aviary_input(self, Aircraft.HorizontalTail.AREA, units='ft**2')
        add_aviary_input(self, Aircraft.HorizontalTail.ASPECT_RATIO, units='unitless')
        # add_aviary_input(self, Aircraft.HorizontalTail.LAMINAR_FLOW_LOWER, 0.0)
        # add_aviary_input(self, Aircraft.HorizontalTail.LAMINAR_FLOW_UPPER, 0.0)
        add_aviary_input(self, Aircraft.HorizontalTail.THICKNESS_TO_CHORD, units='unitless')

        add_aviary_input(self, Aircraft.Nacelle.AVG_DIAMETER, shape=num_engine_type, units='ft')
        add_aviary_input(self, Aircraft.Nacelle.AVG_LENGTH, shape=num_engine_type, units='ft')
        # add_aviary_input(self, Aircraft.Nacelle.LAMINAR_FLOW_LOWER, 0.0)
        # add_aviary_input(self, Aircraft.Nacelle.LAMINAR_FLOW_UPPER, 0.0)

        add_aviary_input(self, Aircraft.VerticalTail.AREA, units='ft**2')
        add_aviary_input(self, Aircraft.VerticalTail.ASPECT_RATIO, units='unitless')
        # add_aviary_input(self, Aircraft.VerticalTail.LAMINAR_FLOW_LOWER, 0.0)
        # add_aviary_input(self, Aircraft.VerticalTail.LAMINAR_FLOW_UPPER, 0.0)
        add_aviary_input(self, Aircraft.VerticalTail.THICKNESS_TO_CHORD, units='unitless')

        add_aviary_output(self, Aircraft.Canard.CHARACTERISTIC_LENGTH, units='ft')
        add_aviary_output(self, Aircraft.Canard.FINENESS, units='unitless')

        add_aviary_output(self, Aircraft.Fuselage.CHARACTERISTIC_LENGTH, units='ft')
        add_aviary_output(self, Aircraft.Fuselage.FINENESS, units='unitless')

        add_aviary_output(self, Aircraft.HorizontalTail.CHARACTERISTIC_LENGTH, units='ft')
        add_aviary_output(self, Aircraft.HorizontalTail.FINENESS, units='unitless')

        add_aviary_output(
            self, Aircraft.Nacelle.CHARACTERISTIC_LENGTH, shape=num_engine_type, units='ft'
        )
        add_aviary_output(self, Aircraft.Nacelle.FINENESS, shape=num_engine_type, units='unitless')

        add_aviary_output(self, Aircraft.VerticalTail.CHARACTERISTIC_LENGTH, units='ft')
        add_aviary_output(self, Aircraft.VerticalTail.FINENESS, units='unitless')

    def setup_partials(self):
        self._setup_partials_horizontal_tail()
        self._setup_partials_vertical_tail()
        self._setup_partials_fuselage()
        self._setup_partials_nacelles()
        self._setup_partials_canard()

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        self._compute_horizontal_tail(inputs, outputs, discrete_inputs, discrete_outputs)

        self._compute_vertical_tail(inputs, outputs, discrete_inputs, discrete_outputs)

        self._compute_fuselage(inputs, outputs, discrete_inputs, discrete_outputs)

        self._compute_nacelles(inputs, outputs, discrete_inputs, discrete_outputs)

        # self._compute_additional_fuselages(
        #     inputs, outputs, discrete_inputs, discrete_outputs
        # )

        # self._compute_additional_vertical_tails(
        #     inputs, outputs, discrete_inputs, discrete_outputs
        # )

        self._compute_canard(inputs, outputs, discrete_inputs, discrete_outputs)

    def compute_partials(self, inputs, J, discrete_inputs=None):
        self._compute_partials_horizontal_tail(inputs, J, discrete_inputs)
        self._compute_partials_vertical_tail(inputs, J, discrete_inputs)
        self._compute_partials_fuselage(inputs, J, discrete_inputs)
        self._compute_partials_nacelles(inputs, J, discrete_inputs)
        self._compute_partials_canard(inputs, J, discrete_inputs)

    def _setup_partials_horizontal_tail(self):
        self.declare_partials(
            Aircraft.HorizontalTail.CHARACTERISTIC_LENGTH,
            [
                Aircraft.HorizontalTail.AREA,
                Aircraft.HorizontalTail.ASPECT_RATIO,
            ],
        )

        self.declare_partials(
            Aircraft.HorizontalTail.FINENESS, Aircraft.HorizontalTail.THICKNESS_TO_CHORD, val=1.0
        )

    def _setup_partials_vertical_tail(self):
        self.declare_partials(
            Aircraft.VerticalTail.CHARACTERISTIC_LENGTH,
            [
                Aircraft.VerticalTail.AREA,
                Aircraft.VerticalTail.ASPECT_RATIO,
            ],
        )

        self.declare_partials(
            Aircraft.VerticalTail.FINENESS, Aircraft.VerticalTail.THICKNESS_TO_CHORD, val=1.0
        )

    def _setup_partials_fuselage(self):
        self.declare_partials(
            Aircraft.Fuselage.CHARACTERISTIC_LENGTH, Aircraft.Fuselage.LENGTH, val=1.0
        )

        self.declare_partials(
            Aircraft.Fuselage.FINENESS,
            [
                Aircraft.Fuselage.AVG_DIAMETER,
                Aircraft.Fuselage.LENGTH,
            ],
        )

    def _setup_partials_nacelles(self):
        # derivatives w.r.t vectorized engine inputs have known sparsity pattern
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])
        shape = np.arange(num_engine_type)

        self.declare_partials(
            Aircraft.Nacelle.CHARACTERISTIC_LENGTH,
            Aircraft.Nacelle.AVG_LENGTH,
            rows=shape,
            cols=shape,
            val=1.0,
        )

        self.declare_partials(
            Aircraft.Nacelle.FINENESS,
            [
                Aircraft.Nacelle.AVG_DIAMETER,
                Aircraft.Nacelle.AVG_LENGTH,
            ],
            rows=shape,
            cols=shape,
            val=1.0,
        )

    def _setup_partials_canard(self):
        self.declare_partials(
            Aircraft.Canard.CHARACTERISTIC_LENGTH,
            [
                Aircraft.Canard.AREA,
                Aircraft.Canard.ASPECT_RATIO,
            ],
        )

        self.declare_partials(
            Aircraft.Canard.FINENESS,
            Aircraft.Canard.THICKNESS_TO_CHORD,
        )

    def _compute_horizontal_tail(
        self, inputs, outputs, discrete_inputs=None, discrete_outputs=None
    ):
        aspect_ratio = inputs[Aircraft.HorizontalTail.ASPECT_RATIO]

        length = 0.0

        if 0.0 < aspect_ratio:
            area = inputs[Aircraft.HorizontalTail.AREA]

            length = (area / aspect_ratio) ** 0.5

        outputs[Aircraft.HorizontalTail.CHARACTERISTIC_LENGTH] = length

        thickness_to_chord = inputs[Aircraft.HorizontalTail.THICKNESS_TO_CHORD]

        outputs[Aircraft.HorizontalTail.FINENESS] = thickness_to_chord

    def _compute_vertical_tail(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        aspect_ratio = inputs[Aircraft.VerticalTail.ASPECT_RATIO]

        length = 0.0

        if 0.0 < aspect_ratio:
            area = inputs[Aircraft.VerticalTail.AREA]

            length = (area / aspect_ratio) ** 0.5

        outputs[Aircraft.VerticalTail.CHARACTERISTIC_LENGTH] = length

        thickness_to_chord = inputs[Aircraft.VerticalTail.THICKNESS_TO_CHORD]

        outputs[Aircraft.VerticalTail.FINENESS] = thickness_to_chord

    def _compute_fuselage(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        length = inputs[Aircraft.Fuselage.LENGTH]

        outputs[Aircraft.Fuselage.CHARACTERISTIC_LENGTH] = length

        avg_diam = inputs[Aircraft.Fuselage.AVG_DIAMETER]

        fineness = length / avg_diam

        outputs[Aircraft.Fuselage.FINENESS] = fineness

    def _compute_nacelles(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # TODO do all engines support nacelles? If not, is this deliberate, or
        # just an artifact of the implementation?
        num_eng = self.options[Aircraft.Engine.NUM_ENGINES]

        avg_diam = inputs[Aircraft.Nacelle.AVG_DIAMETER]
        avg_length = inputs[Aircraft.Nacelle.AVG_LENGTH]

        char_len = np.zeros(len(num_eng), dtype=avg_diam.dtype)
        fineness = np.zeros(len(num_eng), dtype=avg_diam.dtype)

        num_idx = np.where(num_eng >= 1)
        char_len[num_idx] = avg_length[num_idx]
        fineness[num_idx] = 1.0

        calc_idx = np.intersect1d(np.where(avg_diam[num_idx] > 0), num_idx)

        fineness[calc_idx] = avg_length[calc_idx] / avg_diam[calc_idx]

        outputs[Aircraft.Nacelle.CHARACTERISTIC_LENGTH] = char_len
        outputs[Aircraft.Nacelle.FINENESS] = fineness

    # NOTE this code is currently unused!!
    def _compute_additional_fuselages(
        self, inputs, outputs, discrete_inputs=None, discrete_outputs=None
    ):
        num_fuselages = inputs[Aircraft.Fuselage.NUM_FUSELAGES]

        if num_fuselages < 2:
            return

        num_extra = num_fuselages - 1

        idx = self._num_components
        self._num_components += num_extra

        lengths = outputs[Aircraft.Design.CHARACTERISTIC_LENGTHS]

        fineness = outputs[Aircraft.Design.FINENESS]

        laminar_flow_lower = outputs[Aircraft.Design.LAMINAR_FLOW_LOWER]
        laminar_flow_upper = outputs[Aircraft.Design.LAMINAR_FLOW_UPPER]

        for _ in range(num_extra):
            lengths[idx] = lengths[3]

            fineness[idx] = fineness[3]

            laminar_flow_lower[idx] = laminar_flow_lower[3]
            laminar_flow_upper[idx] = laminar_flow_upper[3]

            idx += 1

    # NOTE this code is currently unused!!
    def _compute_additional_vertical_tails(
        self, inputs, outputs, discrete_inputs=None, discrete_outputs=None
    ):
        aviary_options: AviaryValues = self.options['aviary_options']
        num_tails = aviary_options.get_val(Aircraft.VerticalTail.NUM_TAILS)

        if num_tails < 2:
            return

        num_extra = num_tails - 1

        idx = self._num_components
        self._num_components += num_extra

        lengths = outputs[Aircraft.Design.CHARACTERISTIC_LENGTHS]

        fineness = outputs[Aircraft.Design.FINENESS]

        laminar_flow_lower = outputs[Aircraft.Design.LAMINAR_FLOW_LOWER]
        laminar_flow_upper = outputs[Aircraft.Design.LAMINAR_FLOW_UPPER]

        for _ in range(num_extra):
            lengths[idx] = lengths[2]

            fineness[idx] = fineness[2]

            laminar_flow_lower[idx] = laminar_flow_lower[2]
            laminar_flow_upper[idx] = laminar_flow_upper[2]

            idx += 1

    def _compute_canard(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        area = inputs[Aircraft.Canard.AREA]

        if area <= 0.0:
            return

        thickness_to_chord = inputs[Aircraft.Canard.THICKNESS_TO_CHORD]
        aspect_ratio = inputs[Aircraft.Canard.ASPECT_RATIO]

        length = 0.0

        if 0.0 < aspect_ratio:
            length = (area / aspect_ratio) ** 0.5

        outputs[Aircraft.Canard.CHARACTERISTIC_LENGTH] = length

        outputs[Aircraft.Canard.FINENESS] = thickness_to_chord

    def _compute_partials_horizontal_tail(self, inputs, J, discrete_inputs=None):
        aspect_ratio = inputs[Aircraft.HorizontalTail.ASPECT_RATIO]

        da = dr = 0.0

        if 0.0 < aspect_ratio:
            area = inputs[Aircraft.HorizontalTail.AREA]

            f = 0.5 * (area / aspect_ratio) ** -0.5
            da = f / aspect_ratio
            dr = -f * area / aspect_ratio**2.0

        J[Aircraft.HorizontalTail.CHARACTERISTIC_LENGTH, Aircraft.HorizontalTail.AREA] = da

        J[Aircraft.HorizontalTail.CHARACTERISTIC_LENGTH, Aircraft.HorizontalTail.ASPECT_RATIO] = dr

    def _compute_partials_vertical_tail(self, inputs, J, discrete_inputs=None):
        aspect_ratio = inputs[Aircraft.VerticalTail.ASPECT_RATIO]

        da = dr = 0.0

        if 0.0 < aspect_ratio:
            area = inputs[Aircraft.VerticalTail.AREA]

            f = 0.5 * (area / aspect_ratio) ** -0.5
            da = f / aspect_ratio
            dr = -f * area / aspect_ratio**2.0

        J[Aircraft.VerticalTail.CHARACTERISTIC_LENGTH, Aircraft.VerticalTail.AREA] = da

        J[Aircraft.VerticalTail.CHARACTERISTIC_LENGTH, Aircraft.VerticalTail.ASPECT_RATIO] = dr

    def _compute_partials_fuselage(self, inputs, J, discrete_inputs=None):
        length = inputs[Aircraft.Fuselage.LENGTH]
        avg_diam = inputs[Aircraft.Fuselage.AVG_DIAMETER]

        J[Aircraft.Fuselage.FINENESS, Aircraft.Fuselage.LENGTH] = 1.0 / avg_diam

        J[Aircraft.Fuselage.FINENESS, Aircraft.Fuselage.AVG_DIAMETER] = -length / avg_diam**2.0

    def _compute_partials_nacelles(self, inputs, J, discrete_inputs=None):
        num_eng = self.options[Aircraft.Engine.NUM_ENGINES]

        avg_diam = inputs[Aircraft.Nacelle.AVG_DIAMETER]
        avg_length = inputs[Aircraft.Nacelle.AVG_LENGTH]

        avg_diam = inputs[Aircraft.Nacelle.AVG_DIAMETER]
        avg_length = inputs[Aircraft.Nacelle.AVG_LENGTH]

        deriv_char_len = np.zeros(len(num_eng), dtype=avg_diam.dtype)
        deriv_fine_len = np.zeros(len(num_eng), dtype=avg_diam.dtype)
        deriv_fine_diam = np.zeros(len(num_eng), dtype=avg_diam.dtype)

        calc_idx = np.where(num_eng >= 1)
        deriv_char_len[calc_idx] = 1.0
        deriv_fine_len[calc_idx] = 1.0 / avg_diam[calc_idx]
        deriv_fine_diam[calc_idx] = -avg_length[calc_idx] / avg_diam[calc_idx] ** 2.0

        J[Aircraft.Nacelle.CHARACTERISTIC_LENGTH, Aircraft.Nacelle.AVG_LENGTH] = deriv_char_len

        J[Aircraft.Nacelle.FINENESS, Aircraft.Nacelle.AVG_LENGTH] = deriv_fine_len

        J[Aircraft.Nacelle.FINENESS, Aircraft.Nacelle.AVG_DIAMETER] = deriv_fine_diam

    def _compute_partials_canard(self, inputs, J, discrete_inputs=None):
        area = inputs[Aircraft.Canard.AREA]

        if area <= 0.0:
            J[Aircraft.Canard.CHARACTERISTIC_LENGTH, Aircraft.Canard.AREA] = J[
                Aircraft.Canard.CHARACTERISTIC_LENGTH, Aircraft.Canard.ASPECT_RATIO
            ] = J[Aircraft.Canard.FINENESS, Aircraft.Canard.THICKNESS_TO_CHORD] = 0.0

            return

        aspect_ratio = inputs[Aircraft.Canard.ASPECT_RATIO]

        da = dr = 0.0

        if 0.0 < aspect_ratio:
            area = inputs[Aircraft.Canard.AREA]

            f = 0.5 * (area / aspect_ratio) ** -0.5
            da = f / aspect_ratio
            dr = -f * area / aspect_ratio**2.0

        J[Aircraft.Canard.CHARACTERISTIC_LENGTH, Aircraft.Canard.AREA] = da

        J[Aircraft.Canard.CHARACTERISTIC_LENGTH, Aircraft.Canard.ASPECT_RATIO] = dr

        J[Aircraft.Canard.FINENESS, Aircraft.Canard.THICKNESS_TO_CHORD] = 1.0
