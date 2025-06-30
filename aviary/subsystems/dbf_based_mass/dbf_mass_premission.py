import numpy as np
import openmdao.api as om

from aviary.subsystems.dbf_based_mass.dbf_structure_mass import (
    DBFWingMass,
    DBFHorizontalTailMass,
    DBFVerticalTailMass,
)
from aviary.subsystems.dbf_based_mass.dbf_mass_variables import Aircraft


class MassPremission(om.Group):
    def setup(self):
        self.add_subsystem(
            'wing_mass', DBFWingMass(), promotes_inputs=['*'], promotes_outputs=[Aircraft.Wing.MASS]
        )

        self.add_subsystem(
            'horizontal_tail_mass',
            DBFHorizontalTailMass(),
            promotes_inputs=['*'],
            promotes_outputs=[Aircraft.HorizontalTail.MASS],
        )

        self.add_subsystem(
            'vertical_tail_mass',
            DBFVerticalTailMass(),
            promotes_inputs=['*'],
            promotes_outputs=[Aircraft.VerticalTail.MASS],
        )
