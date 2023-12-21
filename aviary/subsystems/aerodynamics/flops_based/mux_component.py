import numpy as np
import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input, get_units
from aviary.variable_info.variables import Aircraft


class MuxComponent(om.ExplicitComponent):
    """
    Assemble component-wise vectors for wetted area, fineness, characteristic length,
    upper-surface laminar fraction, and lower-surface laminar fraction.

    Output ordering matches FLOPS:
    Wing, Horizontal Tail, Vertical Tails, Fuelsages, Nacelles
    """

    def __init__(self, **kwargs):
        self.num_tails = 0
        self.num_fuselages = 0
        self.num_nacelles = 0

        super().__init__(**kwargs)

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        nc = 2
        aviary_options: AviaryValues = self.options['aviary_options']

        # Wing (Always 1)
        add_aviary_input(self, Aircraft.Wing.WETTED_AREA, 1.0)
        add_aviary_input(self, Aircraft.Wing.FINENESS, 1.0)
        add_aviary_input(self, Aircraft.Wing.CHARACTERISTIC_LENGTH, 1.0)
        add_aviary_input(self, Aircraft.Wing.LAMINAR_FLOW_UPPER, 0.0)
        add_aviary_input(self, Aircraft.Wing.LAMINAR_FLOW_LOWER, 0.0)

        # Horizontal Tail (Always 1)
        add_aviary_input(self, Aircraft.HorizontalTail.WETTED_AREA, 1.0)
        add_aviary_input(self, Aircraft.HorizontalTail.FINENESS, 1.0)
        add_aviary_input(self, Aircraft.HorizontalTail.CHARACTERISTIC_LENGTH, 1.0)
        add_aviary_input(self, Aircraft.HorizontalTail.LAMINAR_FLOW_UPPER, 0.0)
        add_aviary_input(self, Aircraft.HorizontalTail.LAMINAR_FLOW_LOWER, 0.0)

        zero_count = (0, None)
        # Vertical Tail
        num, _ = aviary_options.get_item(Aircraft.VerticalTail.NUM_TAILS, zero_count)
        self.num_tails = num
        if num > 0:
            add_aviary_input(self, Aircraft.VerticalTail.WETTED_AREA, 1.0)
            add_aviary_input(self, Aircraft.VerticalTail.FINENESS, 1.0)
            add_aviary_input(self, Aircraft.VerticalTail.CHARACTERISTIC_LENGTH, 1.0)
            add_aviary_input(self, Aircraft.VerticalTail.LAMINAR_FLOW_UPPER, 0.0)
            add_aviary_input(self, Aircraft.VerticalTail.LAMINAR_FLOW_LOWER, 0.0)
            nc += num

        # Fuselage
        num, _ = aviary_options.get_item(Aircraft.Fuselage.NUM_FUSELAGES, zero_count)
        self.num_fuselages = num
        if num > 0:
            add_aviary_input(self, Aircraft.Fuselage.WETTED_AREA, 1.0)
            add_aviary_input(self, Aircraft.Fuselage.FINENESS, 1.0)
            add_aviary_input(self, Aircraft.Fuselage.CHARACTERISTIC_LENGTH, 1.0)
            add_aviary_input(self, Aircraft.Fuselage.LAMINAR_FLOW_UPPER, 0.0)
            add_aviary_input(self, Aircraft.Fuselage.LAMINAR_FLOW_LOWER, 0.0)
            nc += num

        num, _ = aviary_options.get_item(
            Aircraft.Engine.NUM_ENGINES, zero_count)
        self.num_nacelles = int(sum(num))
        if any(num > 0):
            add_aviary_input(self, Aircraft.Nacelle.WETTED_AREA,
                             np.zeros(len(num)))
            add_aviary_input(self, Aircraft.Nacelle.FINENESS,
                             np.zeros(len(num)))
            add_aviary_input(self, Aircraft.Nacelle.CHARACTERISTIC_LENGTH,
                             np.zeros(len(num)))
            add_aviary_input(self, Aircraft.Nacelle.LAMINAR_FLOW_UPPER,
                             np.zeros(len(num)))
            add_aviary_input(self, Aircraft.Nacelle.LAMINAR_FLOW_LOWER,
                             np.zeros(len(num)))
            nc += self.num_nacelles

        self.add_output(
            'wetted_areas', shape=nc,
            units=get_units(Aircraft.Wing.WETTED_AREA))
        self.add_output(
            'fineness_ratios', shape=nc,
            units=get_units(Aircraft.Wing.FINENESS))
        self.add_output(
            'characteristic_lengths', shape=nc,
            units=get_units(Aircraft.Wing.CHARACTERISTIC_LENGTH))
        self.add_output(
            'laminar_fractions_upper', shape=nc,
            units=get_units(Aircraft.Wing.LAMINAR_FLOW_UPPER))
        self.add_output(
            'laminar_fractions_lower', shape=nc,
            units=get_units(Aircraft.Wing.LAMINAR_FLOW_LOWER))

    def setup_partials(self):
        rows = np.zeros(1)
        cols = np.zeros(1)

        # Wing
        self.declare_partials(
            'wetted_areas', Aircraft.Wing.WETTED_AREA,
            rows=rows, cols=cols, val=1.0)
        self.declare_partials(
            'fineness_ratios', Aircraft.Wing.FINENESS,
            rows=rows, cols=cols, val=1.0)
        self.declare_partials(
            'characteristic_lengths', Aircraft.Wing.CHARACTERISTIC_LENGTH,
            rows=rows, cols=cols, val=1.0)
        self.declare_partials(
            'laminar_fractions_upper', Aircraft.Wing.LAMINAR_FLOW_UPPER,
            rows=rows, cols=cols, val=1.0)
        self.declare_partials(
            'laminar_fractions_lower', Aircraft.Wing.LAMINAR_FLOW_LOWER,
            rows=rows, cols=cols, val=1.0)

        # Horizontal Tail
        rows = np.ones(1)
        self.declare_partials(
            'wetted_areas', Aircraft.HorizontalTail.WETTED_AREA,
            rows=rows, cols=cols, val=1.0)
        self.declare_partials(
            'fineness_ratios', Aircraft.HorizontalTail.FINENESS,
            rows=rows, cols=cols, val=1.0)
        self.declare_partials(
            'characteristic_lengths', Aircraft.HorizontalTail.CHARACTERISTIC_LENGTH,
            rows=rows, cols=cols, val=1.0)
        self.declare_partials(
            'laminar_fractions_upper', Aircraft.HorizontalTail.LAMINAR_FLOW_UPPER,
            rows=rows, cols=cols, val=1.0)
        self.declare_partials(
            'laminar_fractions_lower', Aircraft.HorizontalTail.LAMINAR_FLOW_LOWER,
            rows=rows, cols=cols, val=1.0)

        ic = 2

        # Vertical Tail
        if self.num_tails > 0:
            rows = ic + np.arange(self.num_tails)
            cols = np.zeros(self.num_tails)
            self.declare_partials(
                'wetted_areas', Aircraft.VerticalTail.WETTED_AREA,
                rows=rows, cols=cols, val=1.0)
            self.declare_partials(
                'fineness_ratios', Aircraft.VerticalTail.FINENESS,
                rows=rows, cols=cols, val=1.0)
            self.declare_partials(
                'characteristic_lengths', Aircraft.VerticalTail.CHARACTERISTIC_LENGTH,
                rows=rows, cols=cols, val=1.0)
            self.declare_partials(
                'laminar_fractions_upper', Aircraft.VerticalTail.LAMINAR_FLOW_UPPER,
                rows=rows, cols=cols, val=1.0)
            self.declare_partials(
                'laminar_fractions_lower', Aircraft.VerticalTail.LAMINAR_FLOW_LOWER,
                rows=rows, cols=cols, val=1.0)
            ic += self.num_tails

        # Fuselage
        if self.num_fuselages > 0:
            rows = ic + np.arange(self.num_fuselages)
            cols = np.zeros(self.num_fuselages)
            self.declare_partials(
                'wetted_areas', Aircraft.Fuselage.WETTED_AREA,
                rows=rows, cols=cols, val=1.0)
            self.declare_partials(
                'fineness_ratios', Aircraft.Fuselage.FINENESS,
                rows=rows, cols=cols, val=1.0)
            self.declare_partials(
                'characteristic_lengths', Aircraft.Fuselage.CHARACTERISTIC_LENGTH,
                rows=rows, cols=cols, val=1.0)
            self.declare_partials(
                'laminar_fractions_upper', Aircraft.Fuselage.LAMINAR_FLOW_UPPER,
                rows=rows, cols=cols, val=1.0)
            self.declare_partials(
                'laminar_fractions_lower', Aircraft.Fuselage.LAMINAR_FLOW_LOWER,
                rows=rows, cols=cols, val=1.0)
            ic += self.num_fuselages

        # Nacelle
        if self.num_nacelles > 0:
            rows = ic + np.arange(self.num_nacelles)
            cols = np.zeros(self.num_nacelles)
            self.declare_partials(
                'wetted_areas', Aircraft.Nacelle.WETTED_AREA,
                rows=rows, cols=cols, val=1.0)
            self.declare_partials(
                'fineness_ratios', Aircraft.Nacelle.FINENESS,
                rows=rows, cols=cols, val=1.0)
            self.declare_partials(
                'characteristic_lengths', Aircraft.Nacelle.CHARACTERISTIC_LENGTH,
                rows=rows, cols=cols, val=1.0)
            self.declare_partials(
                'laminar_fractions_upper', Aircraft.Nacelle.LAMINAR_FLOW_UPPER,
                rows=rows, cols=cols, val=1.0)
            self.declare_partials(
                'laminar_fractions_lower', Aircraft.Nacelle.LAMINAR_FLOW_LOWER,
                rows=rows, cols=cols, val=1.0)

    def compute(self, inputs, outputs):
        # Wing
        outputs['wetted_areas'][0] = inputs[
            Aircraft.Wing.WETTED_AREA]
        outputs['fineness_ratios'][0] = inputs[
            Aircraft.Wing.FINENESS]
        outputs['characteristic_lengths'][0] = inputs[
            Aircraft.Wing.CHARACTERISTIC_LENGTH]
        outputs['laminar_fractions_upper'][0] = inputs[
            Aircraft.Wing.LAMINAR_FLOW_UPPER]
        outputs['laminar_fractions_lower'][0] = inputs[
            Aircraft.Wing.LAMINAR_FLOW_LOWER]

        # Horizontal Tail
        outputs['wetted_areas'][1] = inputs[
            Aircraft.HorizontalTail.WETTED_AREA]
        outputs['fineness_ratios'][1] = inputs[
            Aircraft.HorizontalTail.FINENESS]
        outputs['characteristic_lengths'][1] = inputs[
            Aircraft.HorizontalTail.CHARACTERISTIC_LENGTH]
        outputs['laminar_fractions_upper'][1] = inputs[
            Aircraft.HorizontalTail.LAMINAR_FLOW_UPPER]
        outputs['laminar_fractions_lower'][1] = inputs[
            Aircraft.HorizontalTail.LAMINAR_FLOW_LOWER]

        ic = 2

        # Vertical Tail
        if self.num_tails > 0:
            for j in range(self.num_tails):
                outputs['wetted_areas'][ic + j] = inputs[
                    Aircraft.VerticalTail.WETTED_AREA]
                outputs['fineness_ratios'][ic + j] = inputs[
                    Aircraft.VerticalTail.FINENESS]
                outputs['characteristic_lengths'][ic + j] = inputs[
                    Aircraft.VerticalTail.CHARACTERISTIC_LENGTH]
                outputs['laminar_fractions_upper'][ic + j] = inputs[
                    Aircraft.VerticalTail.LAMINAR_FLOW_UPPER]
                outputs['laminar_fractions_lower'][ic + j] = inputs[
                    Aircraft.VerticalTail.LAMINAR_FLOW_LOWER]
            ic += self.num_tails

        # Fuselage
        if self.num_fuselages > 0:
            for j in range(self.num_fuselages):
                outputs['wetted_areas'][ic + j] = inputs[Aircraft.Fuselage.WETTED_AREA]
                outputs['fineness_ratios'][ic + j] = inputs[Aircraft.Fuselage.FINENESS]
                outputs['characteristic_lengths'][ic + j] = inputs[
                    Aircraft.Fuselage.CHARACTERISTIC_LENGTH]
                outputs['laminar_fractions_upper'][ic + j] = inputs[
                    Aircraft.Fuselage.LAMINAR_FLOW_UPPER]
                outputs['laminar_fractions_lower'][ic + j] = inputs[
                    Aircraft.Fuselage.LAMINAR_FLOW_LOWER]
            ic += self.num_fuselages

        # Nacelle
        num_engines = self.options['aviary_options'].get_val(Aircraft.Engine.NUM_ENGINES)

        wetted_areas = inputs[Aircraft.Nacelle.WETTED_AREA]
        fineness = inputs[Aircraft.Nacelle.FINENESS]
        char_len = inputs[Aircraft.Nacelle.CHARACTERISTIC_LENGTH]
        lam_up = inputs[Aircraft.Nacelle.LAMINAR_FLOW_UPPER]
        lam_low = inputs[Aircraft.Nacelle.LAMINAR_FLOW_LOWER]

        area_list = np.zeros(0, dtype=wetted_areas.dtype)
        fineness_list = np.zeros(0, dtype=fineness.dtype)
        len_list = np.zeros(0, dtype=char_len.dtype)
        lam_up_list = np.zeros(0, dtype=lam_up.dtype)
        lam_low_list = np.zeros(0, dtype=lam_low.dtype)

        for i, num in enumerate(num_engines):
            if num > 0:
                area_list = np.append(area_list, np.tile(wetted_areas[i], num))
                fineness_list = np.append(fineness_list, np.tile(fineness[i], num))
                len_list = np.append(len_list, np.tile(char_len[i], num))
                lam_up_list = np.append(lam_up_list, np.tile(lam_up[i], num))
                lam_low_list = np.append(lam_low_list, np.tile(lam_low[i], num))

        if self.num_nacelles > 0:
            for j in range(self.num_nacelles):
                outputs['wetted_areas'][ic + j] = area_list[j]
                outputs['fineness_ratios'][ic + j] = fineness_list[j]
                outputs['characteristic_lengths'][ic + j] = len_list[j]
                outputs['laminar_fractions_upper'][ic + j] = lam_up_list[j]
                outputs['laminar_fractions_lower'][ic + j] = lam_low_list[j]
