import openmdao.api as om

from aviary.utils.functions import promote_aircraft_and_mission_vars


class PostMissionGroup(om.Group):
    """OpenMDAO group that holds all post-mission systems."""

    def configure(self):
        """
        Configure this group for post-mission.
        Promote aircraft and mission variables.
        """
        promote_aircraft_and_mission_vars(self)
