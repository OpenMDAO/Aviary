import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import override_aviary_vars
from aviary.variable_info.variable_meta_data import _MetaData


class CorePostMission(om.Group):
    """Group that contains all post-mission groups of core Aviary subsystems: (performance).
    """

    def initialize(self):
        self.options.declare(
            'aviary_options',
            types=AviaryValues,
            desc='collection of Aircraft/Mission specific options',
        )
        self.options.declare('subsystems', desc='list of core subsystem builders')
        self.options.declare('meta_data', desc='problem metadata', default=_MetaData)

        self.options.declare(
            'phase_info',
            desc='The phase_info dict for all phases',
            types=dict,
        )

        self.options.declare(
            'phase_mission_bus_lengths',
            desc='Mapping from phase names to the lengths of the mission_bus_variables timeseries',
            types=dict,
        )

        self.options.declare(
            'post_mission_info',
            desc='The post_mission portion of the phase_info.',
            types=dict,
        )

    def setup(self, **kwargs):
        # rely on openMDAO's auto-ordering for this group
        self.options['auto_order'] = True

        aviary_options = self.options['aviary_options']
        phase_info = self.options['phase_info']
        phase_mission_bus_lengths = self.options['phase_mission_bus_lengths']
        post_mission_info = self.options['post_mission_info']
        core_subsystems = self.options['subsystems']

        for subsystem in core_subsystems:
            pre_mission_system = subsystem.build_post_mission(
                aviary_options,
                phase_info,
                phase_mission_bus_lengths,
                post_mission_info=post_mission_info,
            )
            if pre_mission_system is not None:
                self.add_subsystem(
                    subsystem.name,
                    pre_mission_system,
                    promotes_inputs=['*'],
                    promotes_outputs=['*'],
                )
