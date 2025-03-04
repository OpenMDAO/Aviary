import openmdao.api as om

# Maybe some Aviary inputs as well? 
from aviary.subsystems.mass.simple_mass.wing import WingMassAndCOG
from aviary.subsystems.mass.simple_mass.fuselage import FuselageMassAndCOG
from aviary.subsystems.mass.simple_mass.tail import TailMassAndCOG
from aviary.subsystems.mass.simple_mass.mass_summation import MassSummation

class MassPremission(om.Group):
    """
    Pre-mission group of top-level mass estimation groups and components for 
    the simple small-scale aircraft mass build-up.
    """

    def setup(self):

        self.add_subsystem(
            'Wing',
            WingMassAndCOG(),
            promotes_inputs=['*'], promotes_outputs=['*']
        )

        self.add_subsystem(
            'Fuselage',
            FuselageMassAndCOG(),
            promotes_inputs=['*'], promotes_outputs=['*']
        )

        self.add_subsystem(
            'Tail',
            TailMassAndCOG(),
            promotes_inputs=['*'], promotes_outputs=['*']
        )

        self.add_subsystem(
            'total_mass',
            MassSummation(),
            promotes_inputs=['*'], promotes_outputs=['*']
        )