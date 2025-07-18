import openmdao.api as om

import aviary as av
from aviary.variable_info.variables import Aircraft
from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.examples.external_subsystems.dbf_based_mass.dbf_wing import DBFWingMass
from aviary.examples.external_subsystems.dbf_based_mass.dbf_fuselage import DBFFuselageMass
from aviary.examples.external_subsystems.dbf_based_mass.dbf_horizontaltail import DBFHorizontalTailMass
from aviary.examples.external_subsystems.dbf_based_mass.dbf_verticaltail import DBFVerticalTailMass


class DBFMassBuilder(SubsystemBuilderBase):
    """
    Builder for DBF mass models including wing, horizontal tail, vertical tail, and fuselage.
    """

    def __init__(self, name='dbf_mass'):
        if name is None:
            name = _default_name

        super().__init__(name=name)

    def build_pre_mission(self, aviary_inputs):
        group = om.Group()

        # Add mass subsystems first, promote outputs to group
        group.add_subsystem(
            'wing_mass',
            DBFWingMass(),
            promotes_inputs=['aircraft:*'],
            promotes_outputs=[Aircraft.Wing.MASS],
        )

        group.add_subsystem(
            'horizontal_tail_mass',
            DBFHorizontalTailMass(),
            promotes_inputs=['aircraft:*'],
            promotes_outputs=[Aircraft.HorizontalTail.MASS],
        )

        group.add_subsystem(
            'vertical_tail_mass',
            DBFVerticalTailMass(),
            promotes_inputs=['aircraft:*'],
            promotes_outputs=[Aircraft.VerticalTail.MASS],
        )

        group.add_subsystem(
            'fuselage_mass',
            DBFFuselageMass(),
            promotes_inputs=['aircraft:*'],
            promotes_outputs=[Aircraft.Fuselage.MASS],
        )

        return group

