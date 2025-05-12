import numpy as np

import openmdao.api as om
import openmdao.jax as omj

# Maybe add some aviary inputs at some point here

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

from simple_mass.fuselage import FuselageMassAndCOG
from simple_mass.wing import WingMassAndCOG
from simple_mass.tail import TailMassAndCOG

class MassSummation(om.Group):
    """

    Group to compute various design masses for this mass group.

    This group will be expanded greatly as more subsystems are created. 

    """

    def setup(self):

        self.add_subsystem(
            'fuse_mass', 
            FuselageMassAndCOG(),
            promotes_inputs=['*'],
            promotes_outputs=['total_weight_fuse']
        )

        self.add_subsystem(
            'wing_mass',
            WingMassAndCOG(),
            promotes_inputs=['*'],
            promotes_outputs=['total_weight_wing']
        )

        self.add_subsystem(
            'tail_mass',
            TailMassAndCOG(),
            promotes_inputs=['*'],
            promotes_outputs=['mass']
        )

        self.add_subsystem(
            'structure_mass', 
            StructureMass(),
            promotes_inputs=['*'], 
            promotes_outputs=['*']
        )


class StructureMass(om.JaxExplicitComponent):

    def setup(self):
        # Maybe later change these to Aviary inputs?
        self.add_input('total_weight_wing', val=0.0, units='kg')
        self.add_input('total_weight_fuse', val=0.0, units='kg')
        self.add_input('mass', val=0.0, units='kg')
        # More masses can be added, i.e., tail, spars, flaps, etc. as needed

        self.add_output('structure_mass', val=0.0, units='kg')

    def compute_primal(self, total_weight_wing, total_weight_fuse, mass):
        
        structure_mass = total_weight_wing + total_weight_fuse + mass

        return structure_mass