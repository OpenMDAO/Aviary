import openmdao.api as om

from aviary.subsystems.performance.ode.load_factor import LoadFactor
from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.subsystems.performance.performance_premission import PerformancePremission
from aviary.variable_info.variables import Aircraft, Mission, Dynamic

analysis_dict = {'sustained_load_factor': LoadFactor}
conditions_dict = {
    'altitude': Dynamic.Mission.ALTITUDE,
    'velocity': Dynamic.Mission.VELOCITY,
    'mach': Dynamic.Atmosphere.MACH,
    'throttle': Dynamic.Vehicle.Propulsion.THROTTLE,
    'mass': Dynamic.Vehicle.MASS,
    'fuel_fraction': 'fuel_fraction',
    'delta_weight': 'delta_weight',
    'load_factor': 'load_factor',
}


class CorePerformanceBuilder(SubsystemBuilderBase):
    """Core performance analysis subsystem builder."""

    def __init__(self, name=None, meta_data=None):
        if name is None:
            name = 'core_performance'

        super().__init__(name=name, meta_data=meta_data)

    def build_pre_mission(self, aviary_inputs, **kwargs):
        return PerformancePremission()

    def build_post_mission(self, num_nodes, aviary_inputs, **kwargs):
        if len(kwargs) > 0:
            analyses = [analyses]
            perf_group = om.Group()
        else:
            analyses = []
            perf_group = None

        for perf_analysis in kwargs:
            conditions = kwargs[perf_analysis]
            perf_comp = analysis_dict[perf_analysis]
            # set_input_defaults on perf_comp based on conditions
            perf_group.add_subsystem(name=perf_analysis, subsys=perf_comp(), promotes=['*'])

        return perf_group
