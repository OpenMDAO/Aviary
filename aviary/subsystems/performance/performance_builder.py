from aviary.subsystems.performance.balanced_field_submodel import (
    create_balance_field_subprob,
)
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

    def build_post_mission(self, aviary_inputs, phase_info, phase_mission_bus_lengths, **kwargs):

        if 'post_mission_info' in kwargs:
            post = kwargs['post_mission_info']
            if post.get('balanced_field', False):
                return create_balance_field_subprob(aviary_inputs)

        return