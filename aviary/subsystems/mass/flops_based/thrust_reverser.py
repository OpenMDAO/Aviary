import numpy as np
import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.subsystems.mass.flops_based.distributed_prop import nacelle_count_factor
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft


class ThrustReverserMass(om.ExplicitComponent):
    '''
    Calculates mass of thrust reversers for entire set of an engine model.
    The methodology is based on the FLOPS weight equations, modified to
    output mass instead of weight.

    Assumptions
    -----------
    Total propulsion level thrust is the same as engine set's thrust

    Currently assumed engine set level thrust reversers is the same
    as propulsion-level total sum of thrust reverser masses
    '''

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        engine_count = len(self.options['aviary_options'].get_val(
            Aircraft.Engine.NUM_ENGINES))

        add_aviary_input(
            self, Aircraft.Engine.THRUST_REVERSERS_MASS_SCALER, val=np.zeros(engine_count))
        add_aviary_input(
            self, Aircraft.Engine.SCALED_SLS_THRUST, val=np.zeros(engine_count))

        add_aviary_output(
            self, Aircraft.Engine.THRUST_REVERSERS_MASS, val=np.zeros(engine_count))
        add_aviary_output(
            self, Aircraft.Propulsion.TOTAL_THRUST_REVERSERS_MASS, val=0)

    def setup_partials(self):
        # derivatives w.r.t vectorized engine inputs have known sparsity pattern
        engine_count = len(self.options['aviary_options'].get_val(
            Aircraft.Engine.NUM_ENGINES))
        shape = np.arange(engine_count)

        self.declare_partials(Aircraft.Engine.THRUST_REVERSERS_MASS,
                              Aircraft.Engine.THRUST_REVERSERS_MASS_SCALER,
                              rows=shape, cols=shape, val=1.0)

        self.declare_partials(Aircraft.Engine.THRUST_REVERSERS_MASS,
                              Aircraft.Engine.SCALED_SLS_THRUST,
                              rows=shape, cols=shape, val=1.0)

        self.declare_partials(Aircraft.Propulsion.TOTAL_THRUST_REVERSERS_MASS,
                              Aircraft.Engine.THRUST_REVERSERS_MASS_SCALER,
                              val=1.0)

        self.declare_partials(Aircraft.Propulsion.TOTAL_THRUST_REVERSERS_MASS,
                              Aircraft.Engine.SCALED_SLS_THRUST,
                              val=1.0)

    def compute(self, inputs, outputs):
        aviary_options: AviaryValues = self.options['aviary_options']
        num_eng = aviary_options.get_val(Aircraft.Engine.NUM_ENGINES)
        scaler = inputs[Aircraft.Engine.THRUST_REVERSERS_MASS_SCALER]
        max_thrust = inputs[Aircraft.Engine.SCALED_SLS_THRUST]
        nac_count = nacelle_count_factor(num_eng)

        thrust_reverser_mass = .034 * max_thrust * nac_count * scaler / GRAV_ENGLISH_LBM
        outputs[Aircraft.Engine.THRUST_REVERSERS_MASS] = thrust_reverser_mass
        outputs[Aircraft.Propulsion.TOTAL_THRUST_REVERSERS_MASS] = sum(
            thrust_reverser_mass)

    def compute_partials(self, inputs, J):
        aviary_options: AviaryValues = self.options['aviary_options']
        num_eng = aviary_options.get_val(Aircraft.Engine.NUM_ENGINES)
        scaler = inputs[Aircraft.Engine.THRUST_REVERSERS_MASS_SCALER]
        max_thrust = inputs[Aircraft.Engine.SCALED_SLS_THRUST]
        nac_count = nacelle_count_factor(num_eng)

        J[Aircraft.Engine.THRUST_REVERSERS_MASS,
            Aircraft.Engine.THRUST_REVERSERS_MASS_SCALER] = \
            .034 * max_thrust * nac_count / GRAV_ENGLISH_LBM
        J[Aircraft.Engine.THRUST_REVERSERS_MASS,
            Aircraft.Engine.SCALED_SLS_THRUST] = .034*nac_count * scaler / GRAV_ENGLISH_LBM

        J[Aircraft.Propulsion.TOTAL_THRUST_REVERSERS_MASS,
            Aircraft.Engine.THRUST_REVERSERS_MASS_SCALER] = \
            .034 * max_thrust * nac_count / GRAV_ENGLISH_LBM
        J[Aircraft.Propulsion.TOTAL_THRUST_REVERSERS_MASS,
            Aircraft.Engine.SCALED_SLS_THRUST] = .034*nac_count * scaler / GRAV_ENGLISH_LBM
