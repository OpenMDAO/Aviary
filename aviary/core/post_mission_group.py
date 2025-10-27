import openmdao
import openmdao.api as om
from packaging import version

from aviary.utils.functions import promote_aircraft_and_mission_vars

use_new_openmdao_syntax = version.parse(openmdao.__version__) >= version.parse('3.28')


class PostMissionGroup(om.Group):
    """OpenMDAO group that holds all post-mission systems."""

    def setup(self, **kwargs):
        if use_new_openmdao_syntax:
            # rely on openMDAO's auto-ordering for this group
            self.options['auto_order'] = True

    def configure(self):
        """
        Configure this group for post-mission.
        Promote aircraft and mission variables.
        """
        promote_aircraft_and_mission_vars(self)
