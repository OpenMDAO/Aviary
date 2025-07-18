import openmdao.api as om

from aviary.subsystems.mass.simple_mass.fuselage import FuselageMass
from aviary.subsystems.mass.simple_mass.tail import TailMass
from aviary.subsystems.mass.simple_mass.wing import WingMass


class SimpleMassPremission(om.Group):
    """
    Pre-mission group of top-level mass estimation groups and components for
    the simple small-scale aircraft mass build-up.
    """

    def setup(self):
        self.add_subsystem('Wing', WingMass(), promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem(
            'Fuselage', FuselageMass(), promotes_inputs=['*'], promotes_outputs=['*']
        )

        self.add_subsystem(
            'Tail', TailMass(tail_type='horizontal'), promotes_inputs=['*'], promotes_outputs=['*']
        )
