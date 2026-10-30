import openmdao.api as om

from aviary.utils.functions import promote_aircraft_and_mission_vars
from aviary.variable_info.functions import override_aviary_vars


class StaticGroup(om.Group):
    """Aviary top-level group that is not a dynamic mission.

    This class is used for top pre_mission and post_mission groups.
    """

    def setup(self, **kwargs):
        self.options['auto_order'] = True

    def configure(self):
        """
        Configure this group for pre or post mission.
        Promote aircraft and mission variables.
        Override output aviary variables.
        """
        external_outputs = promote_aircraft_and_mission_vars(self)

        if not hasattr(self, 'core_subsystems'):
            # TODO Post mission doesn't support core subsystems yet.
            return

        core_subs = self.core_subsystems
        override_aviary_vars(
            core_subs,
            core_subs.options['aviary_options'],
            external_overrides=external_outputs,
            code_origin_overrides=core_subs.code_origin_overrides,
        )
