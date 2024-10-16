import openmdao.api as om
from aviary.utils.functions import promote_aircraft_and_mission_vars


class ExternalSubsystemGroup(om.Group):
    def configure(self):
        promote_aircraft_and_mission_vars(self)
