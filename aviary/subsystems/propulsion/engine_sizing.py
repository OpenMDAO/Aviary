import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft


class SizeEngine(om.ExplicitComponent):
    """
    Calculates thrust scaling factors for mission performance parameters. Designed for
    use with EngineDecks.

    Can be vectorized for all unique engines present on aircraft. Each index represents a
    single instance of an engine model.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Engine.REFERENCE_SLS_THRUST, units='lbf')
        add_aviary_option(self, Aircraft.Engine.SCALE_PERFORMANCE)

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
        scale_engine = self.options[Aircraft.Engine.SCALE_PERFORMANCE]
        reference_sls_thrust, _ = self.options[Aircraft.Engine.REFERENCE_SLS_THRUST]

        engine_scale_factor = inputs[Aircraft.Engine.SCALE_FACTOR]

        # Engine is only scaled if required
        # engine scale factor is ratio of scaled thrust target and reference thrust
        if scale_engine:
            scaled_sls_thrust = engine_scale_factor * reference_sls_thrust
        else:
            scaled_sls_thrust = reference_sls_thrust

        outputs[Aircraft.Engine.SCALED_SLS_THRUST] = scaled_sls_thrust

    def setup_partials(self):
        scale_engine = self.options[Aircraft.Engine.SCALE_PERFORMANCE]

        if scale_engine:
            self.declare_partials(Aircraft.Engine.SCALED_SLS_THRUST, Aircraft.Engine.SCALE_FACTOR)

    def compute_partials(self, inputs, J):
        reference_sls_thrust, _ = self.options[Aircraft.Engine.REFERENCE_SLS_THRUST]

        J[Aircraft.Engine.SCALED_SLS_THRUST, Aircraft.Engine.SCALE_FACTOR] = reference_sls_thrust
