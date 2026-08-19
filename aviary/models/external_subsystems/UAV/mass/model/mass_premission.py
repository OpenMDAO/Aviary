import openmdao.api as om
import aviary.api as av
from aviary.models.external_subsystems.UAV.mass.model.wing import WingMass
from aviary.models.external_subsystems.UAV.mass.model.fuselage import FuselageMass
from aviary.models.external_subsystems.UAV.mass.model.horizontaltail import HorizontalTailMass
from aviary.models.external_subsystems.UAV.mass.model.verticaltail import VerticalTailMass
from aviary.models.external_subsystems.UAV.mass.model.mass_summation import MassSummation
from aviary.models.external_subsystems.UAV.UAV_variable_info.UAV_variables import Aircraft

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

