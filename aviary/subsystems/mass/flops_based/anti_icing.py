import numpy as np
import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.subsystems.mass.flops_based.distributed_prop import (
    distributed_engine_count_factor, distributed_nacelle_diam_factor)
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft


class AntiIcingMass(om.ExplicitComponent):
    '''
    Calculates the mass of the anti-icing system. The methodology is based
    on the FLOPS weight equations, modified to output mass instead of weight.
    '''

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(self, Aircraft.AntiIcing.MASS_SCALER, val=1.0)

        add_aviary_input(self, Aircraft.Fuselage.MAX_WIDTH, val=0.0)

        add_aviary_input(self, Aircraft.Nacelle.AVG_DIAMETER, val=0.0)

        add_aviary_input(self, Aircraft.Wing.SPAN, val=0.0)

        add_aviary_input(self, Aircraft.Wing.SWEEP, val=0.0)

        add_aviary_output(self, Aircraft.AntiIcing.MASS, val=0.0)

    def setup_partials(self):
        self.declare_partials("*", "*")

    def compute(self, inputs, outputs):
        aviary_options: AviaryValues = self.options['aviary_options']
        total_engines = aviary_options.get_val(Aircraft.Propulsion.TOTAL_NUM_ENGINES)
        num_engines = aviary_options.get_val(Aircraft.Engine.NUM_ENGINES)

        scaler = inputs[Aircraft.AntiIcing.MASS_SCALER]
        max_width = inputs[Aircraft.Fuselage.MAX_WIDTH]
        avg_diam = inputs[Aircraft.Nacelle.AVG_DIAMETER]
        span = inputs[Aircraft.Wing.SPAN]
        sweep = inputs[Aircraft.Wing.SWEEP]

        count_factor = distributed_engine_count_factor(total_engines)
        f_nacelle = distributed_nacelle_diam_factor(avg_diam, num_engines)

        outputs[Aircraft.AntiIcing.MASS] = (
            (span / np.cos(sweep * np.pi / 180))
            + 3.8 * f_nacelle * count_factor + 1.5 * max_width) * scaler / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):
        aviary_options: AviaryValues = self.options['aviary_options']
        total_engines = aviary_options.get_val(Aircraft.Propulsion.TOTAL_NUM_ENGINES)
        num_engines = aviary_options.get_val(Aircraft.Engine.NUM_ENGINES)

        scaler = inputs[Aircraft.AntiIcing.MASS_SCALER]
        max_width = inputs[Aircraft.Fuselage.MAX_WIDTH]
        avg_diam = inputs[Aircraft.Nacelle.AVG_DIAMETER]
        count_factor = distributed_engine_count_factor(total_engines)
        span = inputs[Aircraft.Wing.SPAN]
        sweep = inputs[Aircraft.Wing.SWEEP]

        weighted_avg_diam = sum(avg_diam * num_engines) / total_engines

        diam_deriv_fact = 1.0
        if total_engines > 4:
            diam_deriv_fact = 0.5 * total_engines ** 0.5

        f_nacelle = weighted_avg_diam * diam_deriv_fact

        cos_sweep = np.cos(sweep * np.pi / 180)
        sin_sweep = np.sin(sweep * np.pi / 180)

        J[Aircraft.AntiIcing.MASS, Aircraft.AntiIcing.MASS_SCALER] = (
            span / cos_sweep + 3.8 * f_nacelle * count_factor +
            1.5 * max_width) / GRAV_ENGLISH_LBM

        J[Aircraft.AntiIcing.MASS, Aircraft.Fuselage.MAX_WIDTH] = \
            1.5 * scaler / GRAV_ENGLISH_LBM

        J[Aircraft.AntiIcing.MASS, Aircraft.Nacelle.AVG_DIAMETER] = \
            3.8 * count_factor * diam_deriv_fact * scaler / GRAV_ENGLISH_LBM

        J[Aircraft.AntiIcing.MASS, Aircraft.Wing.SPAN] = \
            1 / cos_sweep * scaler / GRAV_ENGLISH_LBM

        J[Aircraft.AntiIcing.MASS, Aircraft.Wing.SWEEP] = \
            span * (np.pi / 180) * sin_sweep / (cos_sweep) ** 2 * \
            scaler / GRAV_ENGLISH_LBM
