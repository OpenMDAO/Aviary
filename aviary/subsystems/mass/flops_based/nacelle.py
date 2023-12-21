import numpy as np
import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.subsystems.mass.flops_based.distributed_prop import nacelle_count_factor
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft


class NacelleMass(om.ExplicitComponent):
    '''
    Calculates total mass of all nacelles in a set for an engine model.
    The methodology is based on the FLOPS weight equations, modified to
    output mass instead of weight.

    Assumptions
    -----------
    all engines have identical nacelles, so Nacelle values are being used for propulsion
    -level totals as well as totals for a set of engines for an engine model
    '''

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(self, Aircraft.Nacelle.AVG_DIAMETER, val=np.array([0.0]))

        add_aviary_input(self, Aircraft.Nacelle.AVG_LENGTH, val=np.array([0.0]))

        add_aviary_input(self, Aircraft.Nacelle.MASS_SCALER, val=np.array([1.0]))

        add_aviary_input(self, Aircraft.Engine.SCALED_SLS_THRUST, val=np.array([0.0]))

        add_aviary_output(self, Aircraft.Nacelle.MASS, val=np.array([0.0]))

    def setup_partials(self):
        self.declare_partials("*", "*")

    def compute(self, inputs, outputs):
        aviary_options: AviaryValues = self.options['aviary_options']
        num_eng = aviary_options.get_val(Aircraft.Engine.NUM_ENGINES)
        avg_diam = inputs[Aircraft.Nacelle.AVG_DIAMETER]
        avg_length = inputs[Aircraft.Nacelle.AVG_LENGTH]
        scaler = inputs[Aircraft.Nacelle.MASS_SCALER]

        count_factor = nacelle_count_factor(num_eng)
        # TODO: This should be distributed thrust factor
        thrust = inputs[Aircraft.Engine.SCALED_SLS_THRUST]

        outputs[Aircraft.Nacelle.MASS] = 0.25 * count_factor * \
            avg_diam * avg_length * thrust**0.36 * scaler / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):
        aviary_options: AviaryValues = self.options['aviary_options']
        num_eng = aviary_options.get_val(Aircraft.Engine.NUM_ENGINES)
        avg_diam = inputs[Aircraft.Nacelle.AVG_DIAMETER]
        avg_length = inputs[Aircraft.Nacelle.AVG_LENGTH]
        scaler = inputs[Aircraft.Nacelle.MASS_SCALER]

        count_factor = nacelle_count_factor(num_eng)
        # TODO: This should be distributed thrust factor
        thrust = inputs[Aircraft.Engine.SCALED_SLS_THRUST]

        thrust_exp = thrust**0.36

        J[Aircraft.Nacelle.MASS, Aircraft.Nacelle.AVG_DIAMETER] = 0.25 * \
            count_factor * avg_length * thrust_exp * scaler / GRAV_ENGLISH_LBM
        J[Aircraft.Nacelle.MASS, Aircraft.Nacelle.AVG_LENGTH] = 0.25 * \
            count_factor * avg_diam * thrust_exp * scaler / GRAV_ENGLISH_LBM
        J[Aircraft.Nacelle.MASS, Aircraft.Nacelle.MASS_SCALER] = 0.25 * \
            count_factor * avg_diam * avg_length * thrust_exp / GRAV_ENGLISH_LBM
        J[Aircraft.Nacelle.MASS, Aircraft.Engine.SCALED_SLS_THRUST] = 0.09 * \
            count_factor * avg_diam * avg_length * thrust**-0.64 * scaler / GRAV_ENGLISH_LBM
