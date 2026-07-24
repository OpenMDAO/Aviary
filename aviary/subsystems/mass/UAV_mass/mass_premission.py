import openmdao.api as om
import aviary.api as av
from aviary.subsystems.mass.UAV_mass.wing import WingMass
from aviary.subsystems.mass.UAV_mass.fuselage import FuselageMass
from aviary.subsystems.mass.UAV_mass.horizontaltail import HorizontalTailMass
from aviary.subsystems.mass.UAV_mass.verticaltail import VerticalTailMass
from aviary.subsystems.mass.UAV_mass.mass_summation import MassSummation
from aviary.subsystems.mass.UAV_mass.variable_info.mass_variables import Aircraft

class MassPremission(om.Group):
    def initialize(self):
        self.options.declare("aviary_inputs", types=av.AviaryValues)
        self.options.declare("subsystem_options", types=dict, default={})
    
    def setup(self):
        self.add_subsystem(
            'wing_mass', 
            WingMass(), 
            promotes_inputs=['*'], 
            promotes_outputs=[Aircraft.Wing.MASS],
        )
        self.add_subsystem(
            'horizontal_tail_mass',
            HorizontalTailMass(),
            promotes_inputs=['*'],
            promotes_outputs=[Aircraft.HorizontalTail.MASS],
        )
        self.add_subsystem(
            'vertical_tail_mass',
            VerticalTailMass(),
            promotes_inputs=['*'],
            promotes_outputs=[Aircraft.VerticalTail.MASS],
        )
        self.add_subsystem(
            'fuselage_mass',
            FuselageMass(),
            promotes_inputs=['*'],
            promotes_outputs=[Aircraft.Fuselage.MASS],
        )
        self.add_subsystem(
            'mass_group', 
            MassSummation(), 
            promotes_inputs=['*'], 
            promotes_outputs=['*']
        )
