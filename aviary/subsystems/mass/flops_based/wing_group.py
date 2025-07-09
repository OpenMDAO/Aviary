import openmdao.api as om

from aviary.subsystems.mass.flops_based.engine_pod import EnginePodMass
from aviary.subsystems.mass.flops_based.wing_common import (
    WingBendingMass,
    WingMiscMass,
    WingShearControlMass,
    WingTotalMass,
)
from aviary.subsystems.mass.flops_based.wing_detailed import DetailedWingBendingFact
from aviary.subsystems.mass.flops_based.wing_simple import SimpleWingBendingFact
from aviary.variable_info.functions import add_aviary_option
from aviary.variable_info.variables import Aircraft


class WingMassGroup(om.Group):
    """Group of components used for FLOPS-based wing mass computation."""

    def initialize(self):
        # TODO this requires a special workaround in
        #      variable_info/functions.py, add_aviary_output()
        # default to None instead of default value
        add_aviary_option(self, Aircraft.Wing.USE_DETAILED_MASS)

    def setup(self):
        self.add_subsystem(
            'engine_pod_mass',
            EnginePodMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        if self.options[Aircraft.Wing.USE_DETAILED_MASS]:
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

        self.add_subsystem(
            'wing_total', WingTotalMass(), promotes_inputs=['*'], promotes_outputs=['*']
        )
