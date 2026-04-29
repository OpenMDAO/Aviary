from aviary.subsystems.performance.performance_premission import PerformancePremission
from aviary.subsystems.subsystem_builder import SubsystemBuilder


class PerformanceBuilder(SubsystemBuilder):
    """
    Base performance builder.

    Methods
    -------
    __init__(self, name=None, meta_data=None):
        Initializes the PerformanceBuilder object with a given name.
    """

    _default_name = 'performance'


class CorePerformanceBuilder(PerformanceBuilder):
    """Core performance analysis subsystem builder."""

    def __init__(self, name=None, meta_data=None):
        super().__init__(name=name, meta_data=meta_data)

    def build_pre_mission(self, aviary_inputs, subsystem_options=None):
        return PerformancePremission()
