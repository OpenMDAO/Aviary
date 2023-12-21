import numpy as np
import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission


class LandingTakeoffMassRatio(om.ExplicitComponent):
    '''
    Calculate the ratio of maximum landing mass to maximum takeoff gross mass.
    '''

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(self, Mission.Summary.CRUISE_MACH, val=0.0)

        add_aviary_input(self, Mission.Design.RANGE, val=0.0)

        add_aviary_output(self, Aircraft.Design.LANDING_TO_TAKEOFF_MASS_RATIO, val=0.0)

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        cruise_mach = inputs[Mission.Summary.CRUISE_MACH]
        des_range = inputs[Mission.Design.RANGE]

        # cruise factor set by the cruise Mach number
        # (If statement replaced with expression to give continuous derivatives around
        # Mach=1.0)
        cruise_factor = 5e-5 / (1. + np.exp(-1000 * (cruise_mach - 1))) + 4e-5

        outputs[Aircraft.Design.LANDING_TO_TAKEOFF_MASS_RATIO] = 1 - \
            cruise_factor * des_range

    def compute_partials(self, inputs, J):
        cruise_mach = inputs[Mission.Summary.CRUISE_MACH]
        des_range = inputs[Mission.Design.RANGE]

        den = (1. + np.exp(-1000 * (cruise_mach - 1)))
        cruise_factor = 5e-5 / den + 4e-5
        dfact_dmach = 5e-2 / den ** 2 * np.exp(-1000 * (cruise_mach - 1))

        J[Aircraft.Design.LANDING_TO_TAKEOFF_MASS_RATIO, Mission.Design.RANGE] = \
            -cruise_factor
        J[
            Aircraft.Design.LANDING_TO_TAKEOFF_MASS_RATIO,
            Mission.Summary.CRUISE_MACH] = -des_range * dfact_dmach


class LandingMass(om.ExplicitComponent):
    '''
    Maximum landing mass is maximum takeoff gross mass times the ratio of landing/takeoff mass.
    '''

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(self, Mission.Design.GROSS_MASS, val=0.0)

        add_aviary_input(self, Aircraft.Design.LANDING_TO_TAKEOFF_MASS_RATIO, val=0.0)

        add_aviary_output(self, Aircraft.Design.TOUCHDOWN_MASS, val=0.0)

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        gross_mass = inputs[Mission.Design.GROSS_MASS]
        landing_to_takeoff_mass_ratio = \
            inputs[Aircraft.Design.LANDING_TO_TAKEOFF_MASS_RATIO]

        outputs[Aircraft.Design.TOUCHDOWN_MASS] = gross_mass * \
            landing_to_takeoff_mass_ratio

    def compute_partials(self, inputs, J):
        gross_mass = inputs[Mission.Design.GROSS_MASS]
        landing_to_takeoff_mass_ratio = \
            inputs[Aircraft.Design.LANDING_TO_TAKEOFF_MASS_RATIO]

        J[Aircraft.Design.TOUCHDOWN_MASS, Mission.Design.GROSS_MASS] = \
            landing_to_takeoff_mass_ratio

        J[
            Aircraft.Design.TOUCHDOWN_MASS,
            Aircraft.Design.LANDING_TO_TAKEOFF_MASS_RATIO] = gross_mass
