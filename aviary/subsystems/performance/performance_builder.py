from aviary.subsystems.performance.performance_premission import PerformancePremission
from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase


class CorePerformanceBuilder(SubsystemBuilderBase):
    """Core performance analysis subsystem builder."""

    def __init__(self, name=None, meta_data=None):
        if name is None:
            name = 'core_performance'

        super().__init__(name=name, meta_data=meta_data)

    def build_pre_mission(self, aviary_inputs, **kwargs):
        return PerformancePremission()
