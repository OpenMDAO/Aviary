import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.subsystems.mass.flops_based.distributed_prop import (
    distributed_engine_count_factor,
    distributed_thrust_factor,
)
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft


class TransportEngineOilMass(om.ExplicitComponent):
    """
    Calculates the mass of engine oil using the transport/general aviation method.
    The methodology is based on the FLOPS weight equations, modified to output mass
    instead of weight.

    Assumptions
    -----------
    Calculates total, propulsion-system level mass of all engine oil

    All engines assumed to use engine oil whose mass follows this equation
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Propulsion.TOTAL_NUM_ENGINES)

    def setup(self):
        add_aviary_input(self, Aircraft.Propulsion.ENGINE_OIL_MASS_SCALER, units='unitless')
        add_aviary_input(self, Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST, units='lbf')

        add_aviary_output(self, Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        scaler = inputs[Aircraft.Propulsion.ENGINE_OIL_MASS_SCALER]
        num_eng = self.options[Aircraft.Propulsion.TOTAL_NUM_ENGINES]
        num_eng_fact = distributed_engine_count_factor(num_eng)
        max_sls_thrust = inputs[Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST]
        thrust_factor = distributed_thrust_factor(max_sls_thrust, num_eng)

        outputs[Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS] = (
            0.082 * num_eng_fact * thrust_factor**0.65 * scaler / GRAV_ENGLISH_LBM
        )

    def compute_partials(self, inputs, J):
        scaler = inputs[Aircraft.Propulsion.ENGINE_OIL_MASS_SCALER]
        num_eng = self.options[Aircraft.Propulsion.TOTAL_NUM_ENGINES]
        num_eng_fact = distributed_engine_count_factor(num_eng)
        max_sls_thrust = inputs[Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST]
        thrust_factor = distributed_thrust_factor(max_sls_thrust, num_eng)

        J[Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS, Aircraft.Propulsion.ENGINE_OIL_MASS_SCALER] = (
            0.082 * num_eng_fact * thrust_factor**0.65 / GRAV_ENGLISH_LBM
        )

        J[
            Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS, Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST
        ] = 0.0533 * thrust_factor**-0.35 * scaler / GRAV_ENGLISH_LBM


class AltEngineOilMass(om.ExplicitComponent):
    """
    Calculates the mass of engine oil using the alternate method.
    The methodology is based on the FLOPS weight equations, modified
    to output mass instead of weight.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_PASSENGERS)

    def setup(self):
        add_aviary_input(self, Aircraft.Propulsion.ENGINE_OIL_MASS_SCALER, units='unitless')

        add_aviary_output(self, Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pax = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]

        scaler = inputs[Aircraft.Propulsion.ENGINE_OIL_MASS_SCALER]

        outputs[Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS] = (
            240.0 * ((pax + 39) // 40) * scaler / GRAV_ENGLISH_LBM
        )

    def compute_partials(self, inputs, J):
        pax = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]

        J[Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS, Aircraft.Propulsion.ENGINE_OIL_MASS_SCALER] = (
            240.0 * ((pax + 39) // 40) / GRAV_ENGLISH_LBM
        )
