import openmdao.api as om

from aviary.subsystems.mass.gasp_based.design_load import DesignLoadGroup, BWBDesignLoadGroup
from aviary.subsystems.mass.gasp_based.equipment_and_useful_load import EquipAndUsefulLoadMassGroup
from aviary.subsystems.mass.gasp_based.fixed import FixedMassGroup
from aviary.subsystems.mass.gasp_based.fuel import FuelMassGroup
from aviary.subsystems.mass.gasp_based.wing import WingMassGroup, BWBWingMassGroup
from aviary.variable_info.enums import AircraftTypes
from aviary.variable_info.functions import add_aviary_option
from aviary.variable_info.variables import Aircraft


class MassPremission(om.Group):
    """Pre-mission mass group for GASP-based mass."""

    def initialize(self):
        add_aviary_option(self, Aircraft.Design.TYPE)

    def setup(self):
        design_type = self.options[Aircraft.Design.TYPE]

        # create the instances of the groups

        if design_type is AircraftTypes.BLENDED_WING_BODY:
            self.add_subsystem(
                'design_load',
                BWBDesignLoadGroup(),
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )
        else:
            self.add_subsystem(
                'design_load',
                DesignLoadGroup(),
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )

        self.add_subsystem(
            'fixed_mass',
            FixedMassGroup(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        self.add_subsystem(
            'equip_and_useful_mass',
            EquipAndUsefulLoadMassGroup(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        if design_type is AircraftTypes.BLENDED_WING_BODY:
            self.add_subsystem(
                'wing_mass',
                BWBWingMassGroup(),
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )
        else:
            self.add_subsystem(
                'wing_mass',
                WingMassGroup(),
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )

        self.add_subsystem(
            'fuel_mass',
            FuelMassGroup(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
