import openmdao.api as om

"""
The little bit of path code below is not important overall. This is for me to test 
within the Docker container and VS Code before I push everything fully to the Github 
repository. These lines can be deleted as things are updated further.

"""

import sys
import os


module_path = os.path.abspath("/home/omdao/Aviary/aviary/subsystems/mass")
if module_path not in sys.path:
    sys.path.append(module_path)

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