import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.subsystems.mass.flops_based.distributed_prop import (
    distributed_engine_count_factor, distributed_thrust_factor)
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft


class TransportEngineCtrlsMass(om.ExplicitComponent):
    '''
    Calculate the estimated mass of the engine controls.

    Use for both traditional and blended-wing-body type transports.

    The methodology is based on the FLOPS weight equations, modified
    to output mass instead of weight.

    Assumptions
    -----------
    Calculates total propulsion-system level mass of all engine controls

    All engines have engine controls that use this equation
    '''

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(
            self, Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST, val=0.0, units='lbf')

        add_aviary_output(
            self, Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS, val=0.0, units='lbm')

    def setup_partials(self):
        self.declare_partials(Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS,
                              [Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST])

    def compute(self, inputs, outputs):
        aviary_options: AviaryValues = self.options['aviary_options']

        num_engines = aviary_options.get_val(Aircraft.Propulsion.TOTAL_NUM_ENGINES)
        num_engines_factor = distributed_engine_count_factor(num_engines)

        max_sls_thrust = inputs[Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST]
        thrust_factor = distributed_thrust_factor(max_sls_thrust, num_engines)

        total_controls_weight = 0.26 * num_engines_factor * thrust_factor**0.5

        outputs[Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS] = \
            total_controls_weight / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J, discrete_inputs=None):
        aviary_options: AviaryValues = self.options['aviary_options']

        num_engines = aviary_options.get_val(Aircraft.Propulsion.TOTAL_NUM_ENGINES)
        max_sls_thrust = inputs[Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST]
        thrust_factor = distributed_thrust_factor(max_sls_thrust, num_engines)

        distributed_thrust_factor_exp = thrust_factor**0.5

        J[
            Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS,
            Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST
        ] = 0.13 / distributed_thrust_factor_exp / GRAV_ENGLISH_LBM
