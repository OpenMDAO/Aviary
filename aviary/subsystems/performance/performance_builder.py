import openmdao.api as om

from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.variable_info.variables import Aircraft, Mission

analysis_dict = {'example': om.Group}


class CorePerformanceBuilder(SubsystemBuilderBase):
    """Core performance analysis subsystem builder."""

    def __init__(self, name=None, meta_data=None):
        if name is None:
            name = 'core_performance'

        super().__init__(name=name, meta_data=meta_data)

    def build_pre_mission(self, aviary_inputs, **kwargs):
        return PerformancePreMission()

    def build_mission(self, num_nodes, aviary_inputs, **kwargs):
        try:
            analyses = kwargs.pop('analyses')
        except KeyError:
            analyses = []
            perf_group = None
        else:
            if not isinstance(analyses, list):
                analyses = [analyses]
                perf_group = om.Group()

        for analysis in analyses:
            conditions = analyses[analysis]
            perf_comp = analysis_dict[analysis](*conditions)
            perf_group.add_subsystem(name=analysis, subsys=perf_comp, promotes=['*'])

        return perf_group
