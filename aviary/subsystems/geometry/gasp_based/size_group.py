import openmdao.api as om

from aviary.subsystems.geometry.gasp_based.electric import CableSize
from aviary.subsystems.geometry.gasp_based.empennage import EmpennageSize
from aviary.subsystems.geometry.gasp_based.engine import EngineSize, BWBEngineSizeGroup
from aviary.subsystems.geometry.gasp_based.fuselage import FuselageGroup, BWBFuselageGroup
from aviary.subsystems.geometry.gasp_based.wing import WingGroup, BWBWingGroup
from aviary.variable_info.enums import AircraftTypes
from aviary.variable_info.functions import add_aviary_option
from aviary.variable_info.variables import Aircraft


class SizeGroup(om.Group):
    """Group to pull together all the different components and subgroups of the SIZE subroutine."""

    def initialize(self):
        add_aviary_option(self, Aircraft.Electrical.HAS_HYBRID_SYSTEM)
        add_aviary_option(self, Aircraft.Design.TYPE)

    def setup(self):
        design_type = self.options[Aircraft.Design.TYPE]

        if design_type is AircraftTypes.BLENDED_WING_BODY:
            self.add_subsystem(
                'fuselage',
                BWBFuselageGroup(),
                promotes_inputs=['aircraft:*'],
                promotes_outputs=['aircraft:*'],
            )
        else:
            self.add_subsystem(
                'fuselage',
                FuselageGroup(),
                promotes_inputs=['aircraft:*'],
                promotes_outputs=['aircraft:*'],
            )

        if design_type is AircraftTypes.BLENDED_WING_BODY:
            self.add_subsystem(
                'wing',
                BWBWingGroup(),
                promotes=['aircraft:*', 'mission:*'],
            )
        else:
            self.add_subsystem(
                'wing',
                WingGroup(),
                promotes=['aircraft:*', 'mission:*'],
            )

        self.add_subsystem(
            'empennage',
            EmpennageSize(),
            promotes=['aircraft:*'],
        )

        if design_type is AircraftTypes.BLENDED_WING_BODY:
            self.add_subsystem(
                'engine',
                BWBEngineSizeGroup(),
                promotes_inputs=['aircraft:*'],
                promotes_outputs=['aircraft:*'],
            )
        else:
            self.add_subsystem(
                'engine',
                EngineSize(),
                promotes_inputs=['aircraft:*'],
                promotes_outputs=['aircraft:*'],
            )

        if self.options[Aircraft.Electrical.HAS_HYBRID_SYSTEM]:
            self.add_subsystem(
                'cable',
                CableSize(),
                promotes_inputs=['aircraft:*'],
                promotes_outputs=['aircraft:*'],
            )

        self.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, units='inch')
