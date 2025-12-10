import openmdao.api as om

from aviary.subsystems.mass.flops_based.engine_pod import EnginePodMass
from aviary.subsystems.mass.flops_based.fuselage import BWBAftBodyMass
from aviary.subsystems.mass.flops_based.wing_common import (
    BWBWingMiscMass,
    WingBendingMass,
    WingMiscMass,
    WingShearControlMass,
    WingTotalMass,
)
from aviary.subsystems.mass.flops_based.wing_detailed import (
    BWBDetailedWingBendingFact,
    DetailedWingBendingFact,
)
from aviary.subsystems.mass.flops_based.wing_simple import SimpleWingBendingFact
from aviary.variable_info.enums import AircraftTypes
from aviary.variable_info.functions import add_aviary_option
from aviary.variable_info.variables import Aircraft


class WingMassGroup(om.Group):
    """Group of components used for FLOPS-based wing mass computation."""

    def initialize(self):
        # TODO this requires a special workaround in
        #      variable_info/functions.py, add_aviary_output()
        # default to None instead of default value
        add_aviary_option(self, Aircraft.Wing.DETAILED_WING)
        add_aviary_option(self, Aircraft.Design.TYPE)

    def setup(self):
        design_type = self.options[Aircraft.Design.TYPE]

        self.add_subsystem(
            'engine_pod_mass',
            EnginePodMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        if self.options[Aircraft.Wing.DETAILED_WING]:
            if design_type == AircraftTypes.BLENDED_WING_BODY:
                self.add_subsystem(
                    'wing_bending_material_factor',
                    BWBDetailedWingBendingFact(),
                    promotes_inputs=['*'],
                    promotes_outputs=['*'],
                )
            else:
                self.add_subsystem(
                    'wing_bending_material_factor',
                    DetailedWingBendingFact(),
                    promotes_inputs=['*'],
                    promotes_outputs=['*'],
                )

        else:
            self.add_subsystem(
                'wing_bending_material_factor',
                SimpleWingBendingFact(),
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )

        if design_type == AircraftTypes.BLENDED_WING_BODY:
            self.add_subsystem(
                'wing_misc', BWBWingMiscMass(), promotes_inputs=['*'], promotes_outputs=['*']
            )
        else:
            self.add_subsystem(
                'wing_misc', WingMiscMass(), promotes_inputs=['*'], promotes_outputs=['*']
            )

        self.add_subsystem(
            'wing_shear_control',
            WingShearControlMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        self.add_subsystem(
            'wing_bending',
            WingBendingMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        if design_type == AircraftTypes.BLENDED_WING_BODY:
            self.add_subsystem(
                'aftbody',
                BWBAftBodyMass(),
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )

        self.add_subsystem(
            'wing_total', WingTotalMass(), promotes_inputs=['*'], promotes_outputs=['*']
        )
