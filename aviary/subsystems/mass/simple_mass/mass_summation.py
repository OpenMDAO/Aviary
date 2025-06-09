import numpy as np

import openmdao.api as om
import openmdao.jax as omj

from aviary.subsystems.mass.simple_mass.fuselage import FuselageMassAndCOG
from aviary.subsystems.mass.simple_mass.wing import WingMassAndCOG
from aviary.subsystems.mass.simple_mass.tail import TailMassAndCOG
from aviary.variable_info.variables import Aircraft
# Maybe add some aviary inputs at some point here

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
            promotes_outputs=[Aircraft.Fuselage.MASS]
        )

        self.add_subsystem(
            'wing_mass',
            WingMassAndCOG(),
            promotes_inputs=['*'],
            promotes_outputs=[Aircraft.Wing.MASS]
        )

        self.add_subsystem(
            'tail_mass',
            TailMassAndCOG(),
            promotes_inputs=['*'],
            promotes_outputs=[Aircraft.HorizontalTail.MASS, Aircraft.VerticalTail.MASS]
        )

        self.add_subsystem(
            'structure_mass', 
            StructureMass(),
            promotes_inputs=['*'], 
            promotes_outputs=['*']
        )

# Horizontal tail only
class StructureMass(om.JaxExplicitComponent):

    def setup(self):
        # Maybe later change these to Aviary inputs?
        self.add_input(Aircraft.Wing.MASS, val=0.0, units='kg', primal_name='total_weight_wing')
        self.add_input(Aircraft.Fuselage.MASS, val=0.0, units='kg', primal_name='total_weight_fuse')

        #tail_type = self.tail_mass.options['tail_type']
        
        #if tail_type == 'horizontal':
        self.add_input(Aircraft.HorizontalTail.MASS, val=0.0, units='kg', primal_name='mass')
        #else:
        self.add_input(Aircraft.HorizontalTail.MASS, val=0.0, units='kg', tags='mass')
        # More masses can be added, i.e., tail, spars, flaps, etc. as needed

        self.add_output('structure_mass', val=0.0, units='kg')

    def compute_primal(self, total_weight_wing, total_weight_fuse, mass):
        
        structure_mass = total_weight_wing + total_weight_fuse + mass

        return structure_mass