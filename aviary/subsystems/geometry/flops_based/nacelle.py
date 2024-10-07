import numpy as np
import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft


# NOTE default values for avg diam & avg length if not defined by user:
#      Aircraft.Nacelle.AVG_LENGTH = 0.07 * sqrt(Aircraft.ENGINE.SCALED_SLS_THRUST)
#      Aircraft.Nacelle.AVG_DIAMETER = 0.04 * sqrt(Aircraft.ENGINE.SCALED_SLS_THRUST)
class Nacelles(om.ExplicitComponent):
    """
    Calculate nacelle wetted area and total wetted area
    """

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        num_engine_type = len(self.options['aviary_options'].get_val(
            Aircraft.Engine.NUM_ENGINES))
        add_aviary_input(self, Aircraft.Nacelle.AVG_DIAMETER,
                         val=np.zeros(num_engine_type))
        add_aviary_input(self, Aircraft.Nacelle.AVG_LENGTH,
                         val=np.zeros(num_engine_type))
        add_aviary_input(self, Aircraft.Nacelle.WETTED_AREA_SCALER,
                         val=np.zeros(num_engine_type))

        add_aviary_output(self, Aircraft.Nacelle.TOTAL_WETTED_AREA, 0.0)
        add_aviary_output(self, Aircraft.Nacelle.WETTED_AREA,
                          val=np.zeros(num_engine_type))

    def setup_partials(self):
        # derivatives w.r.t vectorized engine inputs have known sparsity pattern
        num_engine_type = len(self.options['aviary_options'].get_val(
            Aircraft.Engine.NUM_ENGINES))
        shape = np.arange(num_engine_type)

        self.declare_partials(
            Aircraft.Nacelle.TOTAL_WETTED_AREA,
            [
                Aircraft.Nacelle.AVG_DIAMETER, Aircraft.Nacelle.AVG_LENGTH,
                Aircraft.Nacelle.WETTED_AREA_SCALER,
            ]
        )

        self.declare_partials(
            Aircraft.Nacelle.WETTED_AREA,
            [
                Aircraft.Nacelle.AVG_DIAMETER, Aircraft.Nacelle.AVG_LENGTH,
                Aircraft.Nacelle.WETTED_AREA_SCALER,
            ],
            rows=shape, cols=shape, val=1.0
        )

    def compute(
        self, inputs, outputs, discrete_inputs=None, discrete_outputs=None
    ):
        aviary_options: AviaryValues = self.options['aviary_options']
        # how many of each unique engine type are on the aircraft (array)
        num_engines = aviary_options.get_val(Aircraft.Engine.NUM_ENGINES)
        # how many unique engine types are there (int)
        num_engine_type = len(num_engines)

        avg_diam = inputs[Aircraft.Nacelle.AVG_DIAMETER]
        avg_length = inputs[Aircraft.Nacelle.AVG_LENGTH]
        scaler = inputs[Aircraft.Nacelle.WETTED_AREA_SCALER]

        wetted_area = np.zeros(num_engine_type, dtype=avg_diam.dtype)

        calc_idx = np.where(num_engines >= 1)
        wetted_area[calc_idx] = scaler[calc_idx] * 2.8 * \
            avg_diam[calc_idx] * avg_length[calc_idx]

        outputs[Aircraft.Nacelle.WETTED_AREA] = wetted_area

        total_wetted_area = sum(num_engines * wetted_area)

        outputs[Aircraft.Nacelle.TOTAL_WETTED_AREA] = total_wetted_area

    def compute_partials(self, inputs, J, discrete_inputs=None):
        aviary_options: AviaryValues = self.options['aviary_options']
        num_engines = aviary_options.get_val(Aircraft.Engine.NUM_ENGINES)

        avg_diam = inputs[Aircraft.Nacelle.AVG_DIAMETER]
        avg_length = inputs[Aircraft.Nacelle.AVG_LENGTH]
        scaler = inputs[Aircraft.Nacelle.WETTED_AREA_SCALER]

        deriv_area_len = np.zeros(len(num_engines), dtype=avg_diam.dtype)
        deriv_area_diam = np.zeros(len(num_engines), dtype=avg_diam.dtype)
        deriv_area_scaler = np.zeros(len(num_engines), dtype=avg_diam.dtype)
        deriv_total_len = np.zeros(len(num_engines), dtype=avg_diam.dtype)
        deriv_total_diam = np.zeros(len(num_engines), dtype=avg_diam.dtype)
        deriv_total_scaler = np.zeros(len(num_engines), dtype=avg_diam.dtype)

        area_to_length = 2.8 * avg_diam
        area_to_diam = 2.8 * avg_length
        area_to_scaler = 2.8 * avg_diam * avg_length

        calc_idx = np.where(num_engines >= 1)
        deriv_area_len[calc_idx] = scaler[calc_idx] * area_to_length[calc_idx]
        deriv_area_diam[calc_idx] = scaler[calc_idx] * area_to_diam[calc_idx]
        deriv_area_scaler[calc_idx] = area_to_scaler[calc_idx]
        deriv_total_len[calc_idx] = scaler[calc_idx] * \
            num_engines[calc_idx] * area_to_length[calc_idx]
        deriv_total_diam[calc_idx] = scaler[calc_idx] * \
            num_engines[calc_idx] * area_to_diam[calc_idx]
        deriv_total_scaler[calc_idx] = num_engines[calc_idx] * area_to_scaler[calc_idx]

        J[Aircraft.Nacelle.WETTED_AREA, Aircraft.Nacelle.AVG_LENGTH] = deriv_area_len

        J[Aircraft.Nacelle.WETTED_AREA, Aircraft.Nacelle.AVG_DIAMETER] = deriv_area_diam

        J[Aircraft.Nacelle.WETTED_AREA, Aircraft.Nacelle.WETTED_AREA_SCALER] = deriv_area_scaler

        J[Aircraft.Nacelle.TOTAL_WETTED_AREA, Aircraft.Nacelle.AVG_LENGTH] = deriv_total_len

        J[Aircraft.Nacelle.TOTAL_WETTED_AREA, Aircraft.Nacelle.AVG_DIAMETER] = deriv_total_diam

        J[Aircraft.Nacelle.TOTAL_WETTED_AREA,
            Aircraft.Nacelle.WETTED_AREA_SCALER] = deriv_total_scaler
