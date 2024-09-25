import openmdao.api as om

from aviary.subsystems.mass.flops_based.landing_gear import (
    AltLandingGearMass, LandingGearMass, MainGearLength, NoseGearLength)
from aviary.subsystems.mass.flops_based.landing_mass import (
    LandingMass, LandingTakeoffMassRatio)
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Aircraft


class LandingMassGroup(om.Group):
    """
    Group of landing-related components for FLOPS-based mass:
    LandingTakeoffMassRatio, MainGearLength, NoseGearLength, LandingMass, etc.
    """

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        aviary_options: AviaryValues = self.options['aviary_options']
        alt_mass = aviary_options.get_val(Aircraft.Design.USE_ALT_MASS)

        self.add_subsystem('landing_to_takeoff_mass_ratio',
                           LandingTakeoffMassRatio(aviary_options=aviary_options),
                           promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem('main_landing_gear_length',
                           MainGearLength(aviary_options=aviary_options),
                           promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem('nose_landing_gear_length',
                           NoseGearLength(aviary_options=aviary_options),
                           promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem('landing_mass',
                           LandingMass(aviary_options=aviary_options),
                           promotes_inputs=['*'], promotes_outputs=['*'])

        if alt_mass:
            self.add_subsystem('landing_gear',
                               AltLandingGearMass(aviary_options=aviary_options),
                               promotes_inputs=['*'], promotes_outputs=['*'])
        else:
            self.add_subsystem('landing_gear',
                               LandingGearMass(aviary_options=aviary_options),
                               promotes_inputs=['*'], promotes_outputs=['*'])
