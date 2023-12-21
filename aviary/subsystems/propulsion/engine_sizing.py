import numpy as np
import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft


class SizeEngine(om.ExplicitComponent):
    '''
    Calculates thrust scaling factors for dynamic mission parameters. Designed for use
    with EngineDecks.

    Can be vectorized for all unique engines present on aircraft. Each index represents a
    single instance of an engine model.
    '''

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        # count = len(self.options['aviary_options'].get_val('engine_models'))
        count = 1

        add_aviary_input(self, Aircraft.Engine.SCALED_SLS_THRUST, val=np.zeros(count))

        add_aviary_output(self, Aircraft.Engine.SCALE_FACTOR, val=np.zeros(count))

        # variables that also may require scaling
        # TODO - inlet_weight <input>
        # TODO - inlet_weight_scaling_exp <option>
        # TODO - nozzle_weight
        # TODO - nacelle_avg_length: function of sqrt(scale_factor)
        # TODO - nacelle_avg_diam: function of sqrt(scale_factor)
        # TODO - nacelle_wetted_area: if length, diam get scaled - this should be covered by geom

    def compute(self, inputs, outputs):
        options: AviaryValues = self.options['aviary_options']
        # engine_models = options.get_val('engine_models')
        scale_engine = options.get_val(Aircraft.Engine.SCALE_PERFORMANCE)

        reference_sls_thrust = options.get_val(
            Aircraft.Engine.REFERENCE_SLS_THRUST, units='lbf')

        scaled_sls_thrust = inputs[Aircraft.Engine.SCALED_SLS_THRUST]

        # set a default scaling factor of 1 for each engine
        # nm = len(engine_models)

        # use dtype to make complex safe
        # engine_scale_factor = np.ones(nm, dtype=scaled_sls_thrust.dtype)

        # Engine is only scaled if required
        # engine scale factor is ratio of scaled thrust target and reference thrust
        # scale_idx = np.where(scale_engine)
        # engine_scale_factor[scale_idx] = scaled_sls_thrust[scale_idx] / \
        #     reference_sls_thrust[scale_idx]
        engine_scale_factor = 1
        if scale_engine:
            engine_scale_factor = scaled_sls_thrust / reference_sls_thrust

        outputs[Aircraft.Engine.SCALE_FACTOR] = engine_scale_factor

    def setup_partials(self):
        # count = len(self.options['aviary_options'].get_val('engine_models'))

        # shape = np.arange(count, dtype=int)

        self.declare_partials(Aircraft.Engine.SCALE_FACTOR,
                              Aircraft.Engine.SCALED_SLS_THRUST)
        #   rows=shape, cols=shape)

    def compute_partials(self, inputs, J):
        options: AviaryValues = self.options['aviary_options']
        # engine_models = options.get_val('engine_models')
        scale_engine = options.get_val(Aircraft.Engine.SCALE_PERFORMANCE)
        reference_sls_thrust = options.get_val(
            Aircraft.Engine.REFERENCE_SLS_THRUST, units='lbf')

        # nm = len(engine_models)

        scaled_sls_thrust = inputs[Aircraft.Engine.SCALED_SLS_THRUST]
        # use dtype to make complex safe
        # deriv_scale_factor = np.zeros(nm, dtype=scaled_sls_thrust.dtype)

        # scale_idx = np.where(scale_engine)
        # deriv_scale_factor[scale_idx] = 1.0 / reference_sls_thrust[scale_idx]

        deriv_scale_factor = 0
        if scale_engine:
            deriv_scale_factor = 1.0 / reference_sls_thrust

        J[Aircraft.Engine.SCALE_FACTOR,
            Aircraft.Engine.SCALED_SLS_THRUST] = deriv_scale_factor
