import numpy as np
import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input, add_aviary_output, add_aviary_option
from aviary.variable_info.variables import Aircraft


class SizeEngine(om.ExplicitComponent):
    '''
    Calculates thrust scaling factors for mission performance parameters. Designed for
    use with EngineDecks.

    Can be vectorized for all unique engines present on aircraft. Each index represents a
    single instance of an engine model.
    '''

    def initialize(self):
        add_aviary_option(self, Aircraft.Engine.REFERENCE_SLS_THRUST, units='lbf')
        add_aviary_option(self, Aircraft.Engine.SCALE_PERFORMANCE)

    def setup(self):
        add_aviary_input(self, Aircraft.Engine.SCALED_SLS_THRUST, val=0.0)

        add_aviary_output(self, Aircraft.Engine.SCALE_FACTOR, val=0.0)

        # variables that also may require scaling
        # TODO - inlet_weight <input>
        # TODO - inlet_weight_scaling_exp <option>
        # TODO - nozzle_weight
        # TODO - nacelle_avg_length: function of sqrt(scale_factor)
        # TODO - nacelle_avg_diam: function of sqrt(scale_factor)
        # TODO - nacelle_wetted_area: if length, diam get scaled - this should be covered by geom

    def compute(self, inputs, outputs):

        scale_engine = self.options[Aircraft.Engine.SCALE_PERFORMANCE]
        reference_sls_thrust = self.options[Aircraft.Engine.REFERENCE_SLS_THRUST]

        scaled_sls_thrust = inputs[Aircraft.Engine.SCALED_SLS_THRUST]

        # Engine is only scaled if required
        # engine scale factor is ratio of scaled thrust target and reference thrust
        engine_scale_factor = 1
        if scale_engine:
            engine_scale_factor = scaled_sls_thrust / reference_sls_thrust

        outputs[Aircraft.Engine.SCALE_FACTOR] = engine_scale_factor

    def setup_partials(self):
        self.declare_partials(Aircraft.Engine.SCALE_FACTOR,
                              Aircraft.Engine.SCALED_SLS_THRUST)

    def compute_partials(self, inputs, J):
        scale_engine = self.options[Aircraft.Engine.SCALE_PERFORMANCE]
        reference_sls_thrust = self.options[Aircraft.Engine.REFERENCE_SLS_THRUST]

        deriv_scale_factor = 0
        if scale_engine:
            deriv_scale_factor = 1.0 / reference_sls_thrust

        J[Aircraft.Engine.SCALE_FACTOR,
            Aircraft.Engine.SCALED_SLS_THRUST] = deriv_scale_factor
