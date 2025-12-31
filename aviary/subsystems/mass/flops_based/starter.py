import numpy as np
import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.subsystems.mass.flops_based.distributed_prop import (
    distributed_engine_count_factor,
    distributed_nacelle_diam_factor,
    distributed_nacelle_diam_factor_deriv,
)
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission


class TransportStarterMass(om.ExplicitComponent):
    """
    Calculates total sum of all engine starter masses for the entire propulsion
    system (all engines).  The methodology is based on the
    FLOPS weight equations, modified to output mass instead of weight.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Engine.NUM_ENGINES)
        add_aviary_option(self, Aircraft.Propulsion.TOTAL_NUM_ENGINES)
        add_aviary_option(self, Mission.Constraints.MAX_MACH)

    def setup(self):
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])

        add_aviary_input(self, Aircraft.Nacelle.AVG_DIAMETER, shape=num_engine_type, units='ft')
        add_aviary_input(
            self, Aircraft.Engine.SCALE_FACTOR, shape=num_engine_type, units='unitless'
        )

        add_aviary_output(self, Aircraft.Propulsion.TOTAL_STARTER_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        total_engines = self.options[Aircraft.Propulsion.TOTAL_NUM_ENGINES]
        num_engines = self.options[Aircraft.Engine.NUM_ENGINES]
        max_mach = self.options[Mission.Constraints.MAX_MACH]

        d_nacelle = inputs[Aircraft.Nacelle.AVG_DIAMETER]
        num_engines_factor = distributed_engine_count_factor(total_engines)

        # scale avg_diam by thrust ratio
        thrust_rat = inputs[Aircraft.Engine.SCALE_FACTOR]
        adjusted_d_nacelle = d_nacelle * np.sqrt(thrust_rat)

        f_nacelle = distributed_nacelle_diam_factor(adjusted_d_nacelle, num_engines)

        outputs[Aircraft.Propulsion.TOTAL_STARTER_MASS] = (
            11.0 * num_engines_factor * max_mach**0.32 * f_nacelle**1.6
        ) / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):
        total_engines = self.options[Aircraft.Propulsion.TOTAL_NUM_ENGINES]
        num_engines = self.options[Aircraft.Engine.NUM_ENGINES]
        max_mach = self.options[Mission.Constraints.MAX_MACH]

        d_nacelle = inputs[Aircraft.Nacelle.AVG_DIAMETER]
        eng_count_factor = distributed_engine_count_factor(total_engines)

        # scale avg_diam by thrust ratio
        thrust_rat = inputs[Aircraft.Engine.SCALE_FACTOR]
        adjusted_d_nacelle = d_nacelle * np.sqrt(thrust_rat)

        diam_deriv_fact = distributed_nacelle_diam_factor_deriv(num_engines)
        max_mach_exp = max_mach**0.32
        d_avg = sum(adjusted_d_nacelle * num_engines) / total_engines

        J[Aircraft.Propulsion.TOTAL_STARTER_MASS, Aircraft.Nacelle.AVG_DIAMETER] = (
            11.0
            * 1.6
            * eng_count_factor
            * max_mach_exp
            * diam_deriv_fact**1.6
            * np.sqrt(thrust_rat)
            * d_avg**0.6
        ) / GRAV_ENGLISH_LBM

        J[Aircraft.Propulsion.TOTAL_STARTER_MASS, Aircraft.Engine.SCALE_FACTOR] = (
            17.6
            * eng_count_factor
            * max_mach_exp
            * diam_deriv_fact**1.6
            * d_avg**0.6
            * (d_nacelle * 0.5 / np.sqrt(thrust_rat))
        ) / GRAV_ENGLISH_LBM
