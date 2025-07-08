import openmdao.api as om

from aviary.subsystems.mass.simple_mass.wing import WingMassAndCOG
from aviary.subsystems.mass.simple_mass.fuselage import FuselageMassAndCOG
from aviary.subsystems.mass.simple_mass.tail import TailMassAndCOG

class SimpleMassPremission(om.Group):
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
            TailMassAndCOG(tail_type='horizontal'),
            promotes_inputs=['*'], promotes_outputs=['*']
        )