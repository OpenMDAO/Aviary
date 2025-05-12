import openmdao.api as om

from aviary.subsystems.geometry.gasp_based.electric import CableSize
from aviary.subsystems.geometry.gasp_based.empennage import EmpennageSize
from aviary.subsystems.geometry.gasp_based.engine import EngineSize
from aviary.subsystems.geometry.gasp_based.fuselage import FuselageGroup
from aviary.subsystems.geometry.gasp_based.wing import WingGroup
from aviary.variable_info.functions import add_aviary_option
from aviary.variable_info.variables import Aircraft


class SizeGroup(om.Group):
    """Group to pull together all the different components and subgroups of the SIZE subroutine."""

    def initialize(self):
        add_aviary_option(self, Aircraft.Electrical.HAS_HYBRID_SYSTEM)

    def setup(self):
        self.add_subsystem(
            'fuselage',
            FuselageGroup(),
            promotes_inputs=['aircraft:*'],
            promotes_outputs=['aircraft:*'],
        )

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
