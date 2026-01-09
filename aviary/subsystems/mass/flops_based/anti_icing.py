import numpy as np
import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.subsystems.mass.flops_based.distributed_prop import (
    distributed_engine_count_factor,
    distributed_nacelle_diam_factor,
    distributed_nacelle_diam_factor_deriv,
)
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft


class AntiIcingMass(om.ExplicitComponent):
    """
    Calculates the mass of the anti-icing system. The methodology is based
    on the FLOPS weight equations, modified to output mass instead of weight.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Engine.NUM_ENGINES)
        add_aviary_option(self, Aircraft.Propulsion.TOTAL_NUM_ENGINES)
        add_aviary_option(self, Aircraft.Engine.REFERENCE_SLS_THRUST, units='lbf')

    def setup(self):
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])

        add_aviary_input(self, Aircraft.AntiIcing.MASS_SCALER, units='unitless')
        add_aviary_input(self, Aircraft.Fuselage.MAX_WIDTH, units='ft')
        add_aviary_input(self, Aircraft.Nacelle.AVG_DIAMETER, shape=num_engine_type, units='ft')
        add_aviary_input(self, Aircraft.Wing.SPAN, units='ft')
        add_aviary_input(self, Aircraft.Wing.SWEEP, units='deg')
        add_aviary_input(
            self, Aircraft.Engine.SCALED_SLS_THRUST, shape=num_engine_type, units='lbf'
        )

        add_aviary_output(self, Aircraft.AntiIcing.MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        total_engines = self.options[Aircraft.Propulsion.TOTAL_NUM_ENGINES]
        num_engines = self.options[Aircraft.Engine.NUM_ENGINES]

        scaler = inputs[Aircraft.AntiIcing.MASS_SCALER]
        max_width = inputs[Aircraft.Fuselage.MAX_WIDTH]
        avg_diam = inputs[Aircraft.Nacelle.AVG_DIAMETER]
        span = inputs[Aircraft.Wing.SPAN]
        sweep = inputs[Aircraft.Wing.SWEEP]

        thrust = inputs[Aircraft.Engine.SCALED_SLS_THRUST]
        ref_sls_thrust, _ = self.options[Aircraft.Engine.REFERENCE_SLS_THRUST]
        thrust_rat = thrust / ref_sls_thrust
        adjusted_avg_diam = avg_diam * np.sqrt(thrust_rat)

        count_factor = distributed_engine_count_factor(total_engines)
        f_nacelle = distributed_nacelle_diam_factor(adjusted_avg_diam, num_engines)

        outputs[Aircraft.AntiIcing.MASS] = (
            (
                (span / np.cos(sweep * np.pi / 180))
                + 3.8 * f_nacelle * count_factor
                + 1.5 * max_width
            )
            * scaler
            / GRAV_ENGLISH_LBM
        )

    def compute_partials(self, inputs, J):
        total_engines = self.options[Aircraft.Propulsion.TOTAL_NUM_ENGINES]
        num_engines = self.options[Aircraft.Engine.NUM_ENGINES]

        scaler = inputs[Aircraft.AntiIcing.MASS_SCALER]
        max_width = inputs[Aircraft.Fuselage.MAX_WIDTH]
        avg_diam = inputs[Aircraft.Nacelle.AVG_DIAMETER]
        span = inputs[Aircraft.Wing.SPAN]
        sweep = inputs[Aircraft.Wing.SWEEP]

        # scale avg_diam by thrust ratio
        thrust = inputs[Aircraft.Engine.SCALED_SLS_THRUST]
        ref_sls_thrust, _ = self.options[Aircraft.Engine.REFERENCE_SLS_THRUST]
        thrust_rat = thrust / ref_sls_thrust
        adjusted_avg_diam = avg_diam * np.sqrt(thrust_rat)

        count_factor = distributed_engine_count_factor(total_engines)
        f_nacelle = distributed_nacelle_diam_factor(adjusted_avg_diam, num_engines)

        diam_deriv_fact = distributed_nacelle_diam_factor_deriv(num_engines)

        cos_sweep = np.cos(sweep * np.pi / 180)
        sin_sweep = np.sin(sweep * np.pi / 180)

        J[Aircraft.AntiIcing.MASS, Aircraft.AntiIcing.MASS_SCALER] = (
            span / cos_sweep + 3.8 * f_nacelle * count_factor + 1.5 * max_width
        ) / GRAV_ENGLISH_LBM

        J[Aircraft.AntiIcing.MASS, Aircraft.Fuselage.MAX_WIDTH] = 1.5 * scaler / GRAV_ENGLISH_LBM

        J[Aircraft.AntiIcing.MASS, Aircraft.Nacelle.AVG_DIAMETER] = (
            3.8 * diam_deriv_fact * np.sqrt(thrust_rat) * count_factor * scaler / GRAV_ENGLISH_LBM
        )

        J[Aircraft.AntiIcing.MASS, Aircraft.Wing.SPAN] = 1 / cos_sweep * scaler / GRAV_ENGLISH_LBM

        J[Aircraft.AntiIcing.MASS, Aircraft.Wing.SWEEP] = (
            span * (np.pi / 180) * sin_sweep / (cos_sweep) ** 2 * scaler / GRAV_ENGLISH_LBM
        )

        J[Aircraft.AntiIcing.MASS, Aircraft.Engine.SCALED_SLS_THRUST] = (
            3.8
            * diam_deriv_fact
            * avg_diam
            * 0.5
            * np.sqrt(ref_sls_thrust / thrust)
            / ref_sls_thrust
            * count_factor
            * scaler
            / GRAV_ENGLISH_LBM
        )
