import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.subsystems.mass.flops_based.distributed_prop import (
    distributed_engine_count_factor,
    distributed_nacelle_diam_factor,
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

        add_aviary_output(self, Aircraft.Propulsion.TOTAL_STARTER_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        total_engines = self.options[Aircraft.Propulsion.TOTAL_NUM_ENGINES]
        num_engines = self.options[Aircraft.Engine.NUM_ENGINES]
        max_mach = self.options[Mission.Constraints.MAX_MACH]

        d_nacelle = inputs[Aircraft.Nacelle.AVG_DIAMETER]
        num_engines_factor = distributed_engine_count_factor(total_engines)
        f_nacelle = distributed_nacelle_diam_factor(d_nacelle, num_engines)

        outputs[Aircraft.Propulsion.TOTAL_STARTER_MASS] = (
            11.0 * num_engines_factor * max_mach**0.32 * f_nacelle**1.6
        ) / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):
        total_engines = self.options[Aircraft.Propulsion.TOTAL_NUM_ENGINES]
        num_engines = self.options[Aircraft.Engine.NUM_ENGINES]
        max_mach = self.options[Mission.Constraints.MAX_MACH]

        d_nacelle = inputs[Aircraft.Nacelle.AVG_DIAMETER]
        eng_count_factor = distributed_engine_count_factor(total_engines)

        d_avg = sum(d_nacelle * num_engines) / total_engines

        diam_deriv_fact = 1
        if total_engines > 4:
            diam_deriv_fact = (0.5 * total_engines**0.5) ** 1.6

        max_mach_exp = max_mach**0.32

        J[Aircraft.Propulsion.TOTAL_STARTER_MASS, Aircraft.Nacelle.AVG_DIAMETER] = (
            17.6 * eng_count_factor * max_mach_exp * diam_deriv_fact * d_avg**0.6 / GRAV_ENGLISH_LBM
        )
