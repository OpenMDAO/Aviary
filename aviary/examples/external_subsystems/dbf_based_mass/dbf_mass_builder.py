import openmdao.api as om
import aviary as av
from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.examples.external_subsystems.dbf_based_mass.dbf_mass_variable_meta_data import (
    ExtendedMetaData,
)
from aviary.examples.external_subsystems.dbf_based_mass.dbf_structure_mass import (
    DBFWingMass,
    DBFHorizontalTailMass,
    DBFVerticalTailMass,
    DBFFuselageMass,
)


class DBFMassBuilder(SubsystemBuilderBase):
    """
    Builder for DBF mass models including wing, horizontal tail, vertical tail, and fuselage.
    """

    def __init__(self, name='dbf_mass', meta_data=None):
        super().__init__(name)
        self.meta_data = meta_data

    def build_pre_mission(self, aviary_inputs):
        group = om.Group()

        group.add_subsystem(
            'wing_mass',
            DBFWingMass(),
            promotes_inputs=['aircraft:*'],
            promotes_outputs=['aircraft:wing:mass'],
        )

        group.add_subsystem(
            'horizontal_tail_mass',
            DBFHorizontalTailMass(),
            promotes_inputs=['aircraft:*'],
            promotes_outputs=['aircraft:horizontal_tail:mass'],
        )

        group.add_subsystem(
            'vertical_tail_mass',
            DBFVerticalTailMass(),
            promotes_inputs=['aircraft:*'],
            promotes_outputs=['aircraft:vertical_tail:mass'],
        )

        group.add_subsystem(
            'fuselage_mass',
            DBFFuselageMass(),
            promotes_inputs=['aircraft:*'],
            promotes_outputs=['aircraft:fuselage:mass'],
        )

        return group
