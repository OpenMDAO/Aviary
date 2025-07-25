import numpy as np

import openmdao.api as om


from aviary.examples.external_subsystems.dbf_based_mass.dbf_fuselage import DBFFuselageMass
from aviary.examples.external_subsystems.dbf_based_mass.dbf_verticaltail import DBFVerticalTailMass
from aviary.examples.external_subsystems.dbf_based_mass.dbf_horizontaltail import DBFHorizontalTailMass
from aviary.examples.external_subsystems.dbf_based_mass.dbf_wing import DBFWingMass
from aviary.variable_info.variables import Aircraft
from aviary.variable_info.functions import add_aviary_input


class MassSummation(om.Group):
    """

    Group to compute various design masses for this mass group.

    This group will be expanded greatly as more subsystems are created. 

    """
    def setup(self):
        self.add_subsystem(
            'structure_mass_sum',
            StructureMass(),
            promotes_inputs=[
                Aircraft.Wing.MASS,
                Aircraft.Fuselage.MASS,
                Aircraft.HorizontalTail.MASS,
                Aircraft.VerticalTail.MASS,
            ],
            promotes_outputs=['structure_mass'])

        self.add_subsystem(
            'total_mass_sum',
            TotalMass(),
            promotes_inputs=[
                'structure_mass',
                Aircraft.Battery.MASS,
                Aircraft.Engine.Motor.MASS
            ],
            promotes_outputs=['total_mass'])
        
class StructureMass(om.ExplicitComponent):
    def setup(self):
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

    def compute(self, inputs, outputs):
        outputs['structure_mass'] = (
            inputs[Aircraft.Wing.MASS] +
            inputs[Aircraft.Fuselage.MASS] +
            inputs[Aircraft.HorizontalTail.MASS] +
            inputs[Aircraft.VerticalTail.MASS]
        )

class TotalMass(om.ExplicitComponent):
    def setup(self):
        self.add_input('structure_mass', val=0.0, units='kg')
        add_aviary_input(self, Aircraft.Battery.MASS, val=0.0, units='kg')
        add_aviary_input(self, Aircraft.Engine.Motor.MASS, val=0.0, units='kg')

        self.add_output('total_mass', val=0.0, units='kg')

    def compute(self, inputs, outputs):
        outputs['total_mass'] = (
            inputs['structure_mass'] +
            inputs[Aircraft.Battery.MASS] +
            inputs[Aircraft.Engine.Motor.MASS]
        )