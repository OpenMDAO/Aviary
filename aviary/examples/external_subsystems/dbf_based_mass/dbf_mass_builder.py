import openmdao.api as om
from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
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

    def __init__(self, name='dbf_mass'):
        super().__init__(name)

    def build_pre_mission(self, aviary_inputs):
        group = om.Group()

        group.add_subsystem(
            'wing_mass',
            DBFWingMass(),
            promotes_inputs=['aircraft:*'],
            promotes_outputs=['aircraft:dbfwing:mass'],
        )

        group.add_subsystem(
            'horizontal_tail_mass',
            DBFHorizontalTailMass(),
            promotes_inputs=['aircraft:*'],
            promotes_outputs=['aircraft:dbfhorizontal_tail:mass'],
        )

        group.add_subsystem(
            'vertical_tail_mass',
            DBFVerticalTailMass(),
            promotes_inputs=['aircraft:*'],
            promotes_outputs=['aircraft:dbfVerticalTail:mass'],
        )

        group.add_subsystem(
            'fuselage_mass',
            DBFFuselageMass(),
            promotes_inputs=['aircraft:*'],
            promotes_outputs=['aircraft:dbfFuselage:mass'],
        )

        return group
