import openmdao.api as om

from aviary.subsystems.geometry.gasp_based.electric import CableSize
from aviary.subsystems.geometry.gasp_based.empennage import EmpennageSize
from aviary.subsystems.geometry.gasp_based.engine import EngineSize
from aviary.subsystems.geometry.gasp_based.fuselage import FuselageGroup
from aviary.subsystems.geometry.gasp_based.wing import WingGroup
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Aircraft


class SizeGroup(om.Group):

    """
    Group to pull together all the different components and subgroups of the SIZE subroutine

    """

    def initialize(self):

        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options'
        )

    def setup(self):
        aviary_options = self.options['aviary_options']

        self.add_subsystem(
            "fuselage",
            FuselageGroup(
                aviary_options=aviary_options,
            ),
            promotes_inputs=["aircraft:*"],
            promotes_outputs=["aircraft:*"],
        )

        self.add_subsystem(
            "wing",
            WingGroup(
                aviary_options=aviary_options,
            ),
            promotes=["aircraft:*", "mission:*"],
        )

        self.add_subsystem(
            "empennage",
            EmpennageSize(aviary_options=aviary_options,),
            promotes=["aircraft:*"],
        )

        self.add_subsystem(
            "engine",
            EngineSize(aviary_options=aviary_options,),
            promotes_inputs=["aircraft:*"],
            promotes_outputs=["aircraft:*"],
        )

        if self.options["aviary_options"].get_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, units='unitless'):
            self.add_subsystem(
                "cable",
                CableSize(aviary_options=aviary_options,),
                promotes_inputs=["aircraft:*"],
                promotes_outputs=["aircraft:*"],
            )

        self.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, units="inch")
