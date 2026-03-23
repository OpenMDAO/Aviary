import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues
from aviary.utils.functions import promote_aircraft_and_mission_vars
from aviary.variable_info.functions import override_aviary_vars
from aviary.variable_info.variable_meta_data import _MetaData


class PreMissionGroup(om.Group):
    """OpenMDAO group that holds all pre-mission systems."""

    def initialize(self):
        self.options.declare(
            'aviary_options',
            types=AviaryValues,
            desc='collection of Aircraft/Mission specific options',
        )
        self.options.declare('subsystems', desc='list of core subsystem builders')
        self.options.declare('meta_data', desc='problem metadata', default=_MetaData)

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
            code_origin_overrides=pre_mission.code_origin_overrides,
        )
