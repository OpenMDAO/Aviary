import openmdao.api as om

from aviary.subsystems.mass.flops_based.landing_gear import (
    AltLandingGearMass,
    LandingGearMass,
    MainGearLength,
    NoseGearLength,
)
from aviary.subsystems.mass.flops_based.landing_mass import LandingMass, LandingTakeoffMassRatio
from aviary.variable_info.functions import add_aviary_option
from aviary.variable_info.variables import Aircraft


class LandingMassGroup(om.Group):
    """
    Group of landing-related components for FLOPS-based mass:
    LandingTakeoffMassRatio, MainGearLength, NoseGearLength, LandingMass, etc.
    """

    def setup(self):
        self.add_subsystem(
            'landing_to_takeoff_mass_ratio',
            LandingTakeoffMassRatio(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        self.add_subsystem(
            'main_landing_gear_length',
            MainGearLength(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        self.add_subsystem(
            'nose_landing_gear_length',
            NoseGearLength(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        self.add_subsystem(
            'landing_mass', LandingMass(), promotes_inputs=['*'], promotes_outputs=['*']
        )

        self.add_subsystem(
            'landing_gear', LandingGearMass(), promotes_inputs=['*'], promotes_outputs=['*']
        )
