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

        self.add_subsystem(
            'wing_area_ratios',
            om.ExecComp(
                [
                    'ht_area_ratio = horizontal_tail_area / wing_area',
                    'vt_area_ratio = vertical_tail_area / wing_area',
                ],
                wing_area={'val': 0.0, 'units': 'm**2'},
                horizontal_tail_area={'val': 0.0, 'units': 'm**2'},
                vertical_tail_area={'val': 0.0, 'units': 'm**2'},
                ht_area_ratio={'val': 0.0, 'units': 'unitless'},
                vt_area_ratio={'val': 0.0, 'units': 'unitless'},
            ),
            promotes_inputs=['*'],
            promotes_outputs=['ht_area_ratio', 'vt_area_ratio'],
        )