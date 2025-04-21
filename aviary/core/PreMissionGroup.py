import openmdao.api as om

from aviary.utils.functions import promote_aircraft_and_mission_vars
from aviary.variable_info.functions import override_aviary_vars


class PreMissionGroup(om.Group):
    """OpenMDAO group that holds all pre-mission systems."""

    def configure(self):
        """
        Configure this group for pre-mission.
        Promote aircraft and mission variables.
        Override output aviary variables.
        """
        external_outputs = promote_aircraft_and_mission_vars(self)

        pre_mission = self.core_subsystems
        override_aviary_vars(
            pre_mission,
            pre_mission.options['aviary_options'],
            external_overrides=external_outputs,
            manual_overrides=pre_mission.manual_overrides,
        )
