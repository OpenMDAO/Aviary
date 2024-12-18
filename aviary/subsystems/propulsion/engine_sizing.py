import numpy as np
import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft


class SizeEngine(om.ExplicitComponent):
    '''
    Calculates thrust scaling factors for mission performance parameters. Designed for
    use with EngineDecks.

    Can be vectorized for all unique engines present on aircraft. Each index represents a
    single instance of an engine model.
    '''

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(self, Aircraft.Engine.SCALE_FACTOR, val=1.0)

        add_aviary_output(self, Aircraft.Engine.SCALED_SLS_THRUST, val=0.0)

        # variables that also may require scaling
        # TODO - inlet_weight <input>
        # TODO - inlet_weight_scaling_exp <option>
        # TODO - nozzle_weight
        # TODO - nacelle_avg_length: function of sqrt(scale_factor)
        # TODO - nacelle_avg_diam: function of sqrt(scale_factor)
        # TODO - nacelle_wetted_area: if length, diam get scaled - this should be covered by geom

    def compute(self, inputs, outputs):
        options: AviaryValues = self.options['aviary_options']

        scale_engine = options.get_val(Aircraft.Engine.SCALE_PERFORMANCE)

        reference_sls_thrust = options.get_val(Aircraft.Engine.REFERENCE_SLS_THRUST,
                                               units='lbf')

        engine_scale_factor = inputs[Aircraft.Engine.SCALE_FACTOR]

        # Engine is only scaled if required
        # engine scale factor is ratio of scaled thrust target and reference thrust
        if scale_engine:
            scaled_sls_thrust = engine_scale_factor * reference_sls_thrust
        else:
            scaled_sls_thrust = reference_sls_thrust

        outputs[Aircraft.Engine.SCALED_SLS_THRUST] = scaled_sls_thrust

    def setup_partials(self):
        self.declare_partials(
            Aircraft.Engine.SCALED_SLS_THRUST, Aircraft.Engine.SCALE_FACTOR
        )

    def compute_partials(self, inputs, J):
        options: AviaryValues = self.options['aviary_options']
        reference_sls_thrust = options.get_val(
            Aircraft.Engine.REFERENCE_SLS_THRUST, units='lbf')

        J[Aircraft.Engine.SCALED_SLS_THRUST, Aircraft.Engine.SCALE_FACTOR] = (
            reference_sls_thrust
        )
