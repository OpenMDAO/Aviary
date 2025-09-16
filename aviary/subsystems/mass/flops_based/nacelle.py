import numpy as np
import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.subsystems.mass.flops_based.distributed_prop import nacelle_count_factor
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft


class NacelleMass(om.ExplicitComponent):
    """
    Calculates total mass of all nacelles in a set for an engine model.
    The methodology is based on the FLOPS weight equations, modified to
    output mass instead of weight.

    Assumptions
    -----------
    all engines have identical nacelles, so Nacelle values are being used for propulsion
    -level totals as well as totals for a set of engines for an engine model
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Engine.NUM_ENGINES)

    def setup(self):
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])

        add_aviary_input(self, Aircraft.Nacelle.AVG_DIAMETER, shape=num_engine_type, units='ft')
        add_aviary_input(self, Aircraft.Nacelle.AVG_LENGTH, shape=num_engine_type, units='ft')
        add_aviary_input(
            self, Aircraft.Nacelle.MASS_SCALER, shape=num_engine_type, units='unitless'
        )
        add_aviary_input(
            self, Aircraft.Engine.SCALED_SLS_THRUST, shape=num_engine_type, units='lbf'
        )

        add_aviary_output(self, Aircraft.Nacelle.MASS, shape=num_engine_type, units='lbm')

    def setup_partials(self):
        # derivatives w.r.t vectorized engine inputs have known sparsity pattern
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])
        shape = np.arange(num_engine_type)

        self.declare_partials(
            Aircraft.Nacelle.MASS, Aircraft.Nacelle.AVG_DIAMETER, rows=shape, cols=shape, val=1.0
        )
        self.declare_partials(
            Aircraft.Nacelle.MASS, Aircraft.Nacelle.AVG_LENGTH, rows=shape, cols=shape, val=1.0
        )
        self.declare_partials(
            Aircraft.Nacelle.MASS, Aircraft.Nacelle.MASS_SCALER, rows=shape, cols=shape, val=1.0
        )
        self.declare_partials(
            Aircraft.Nacelle.MASS,
            Aircraft.Engine.SCALED_SLS_THRUST,
            rows=shape,
            cols=shape,
            val=1.0,
        )

    def compute(self, inputs, outputs):
        num_eng = self.options[Aircraft.Engine.NUM_ENGINES]
        avg_diam = inputs[Aircraft.Nacelle.AVG_DIAMETER]
        avg_length = inputs[Aircraft.Nacelle.AVG_LENGTH]
        scaler = inputs[Aircraft.Nacelle.MASS_SCALER]

        count_factor = nacelle_count_factor(num_eng)
        # TODO: This should be distributed thrust factor
        thrust = inputs[Aircraft.Engine.SCALED_SLS_THRUST]

        outputs[Aircraft.Nacelle.MASS] = (
            0.25 * count_factor * avg_diam * avg_length * thrust**0.36 * scaler / GRAV_ENGLISH_LBM
        )

    def compute_partials(self, inputs, J):
        num_eng = self.options[Aircraft.Engine.NUM_ENGINES]
        avg_diam = inputs[Aircraft.Nacelle.AVG_DIAMETER]
        avg_length = inputs[Aircraft.Nacelle.AVG_LENGTH]
        scaler = inputs[Aircraft.Nacelle.MASS_SCALER]

        count_factor = nacelle_count_factor(num_eng)
        # TODO: This should be distributed thrust factor
        thrust = inputs[Aircraft.Engine.SCALED_SLS_THRUST]

        thrust_exp = thrust**0.36

        J[Aircraft.Nacelle.MASS, Aircraft.Nacelle.AVG_DIAMETER] = (
            0.25 * count_factor * avg_length * thrust_exp * scaler / GRAV_ENGLISH_LBM
        )
        J[Aircraft.Nacelle.MASS, Aircraft.Nacelle.AVG_LENGTH] = (
            0.25 * count_factor * avg_diam * thrust_exp * scaler / GRAV_ENGLISH_LBM
        )
        J[Aircraft.Nacelle.MASS, Aircraft.Nacelle.MASS_SCALER] = (
            0.25 * count_factor * avg_diam * avg_length * thrust_exp / GRAV_ENGLISH_LBM
        )
        J[Aircraft.Nacelle.MASS, Aircraft.Engine.SCALED_SLS_THRUST] = (
            0.09 * count_factor * avg_diam * avg_length * thrust**-0.64 * scaler / GRAV_ENGLISH_LBM
        )
