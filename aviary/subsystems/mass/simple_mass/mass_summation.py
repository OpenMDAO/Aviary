import numpy as np

import openmdao.api as om


from aviary.subsystems.mass.simple_mass.fuselage import FuselageMassAndCOG
from aviary.subsystems.mass.simple_mass.wing import WingMassAndCOG
from aviary.subsystems.mass.simple_mass.tail import TailMassAndCOG
from aviary.variable_info.variables import Aircraft
from aviary.variable_info.functions import add_aviary_input


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
            promotes_outputs=[Aircraft.HorizontalTail.MASS]
        )

        self.add_subsystem(
            'structure_mass', 
            StructureMass(),
            promotes_inputs=['*'], 
            promotes_outputs=['*']
        )

class StructureMass(om.JaxExplicitComponent):
    def initialize(self):
        self.options.declare('tail_type', 
                     default='horizontal',
                     values=['horizontal', 'vertical'],
                     desc="Tail type used for the tail mass from tail.py file")

    def setup(self):
        tail_type = self.options['tail_type']

        add_aviary_input(self, 
                         Aircraft.Wing.MASS, 
                         val=0.0, 
                         units='kg')
        
        add_aviary_input(self, 
                         Aircraft.Fuselage.MASS, 
                         val=0.0, 
                         units='kg')

        add_aviary_input(self, 
                         Aircraft.HorizontalTail.MASS, 
                         val=0.0, 
                         units='kg')
        
        add_aviary_input(self, 
                         Aircraft.VerticalTail.MASS, 
                         val=0.0, 
                         units='kg')
        
        # More masses can be added, i.e., tail, spars, flaps, etc. as needed

        self.add_output('structure_mass', 
                        val=0.0, 
                        units='kg')

    def compute_primal(self, aircraft__wing__mass, aircraft__fuselage__mass, aircraft__horizontal_tail__mass, aircraft__vertical_tail__mass):

        tail_type = self.options['tail_type']

        if tail_type == 'horizontal':
            structure_mass = aircraft__wing__mass + aircraft__fuselage__mass + aircraft__horizontal_tail__mass
        else:
            structure_mass = aircraft__wing__mass + aircraft__fuselage__mass + aircraft__vertical_tail__mass

        return structure_mass