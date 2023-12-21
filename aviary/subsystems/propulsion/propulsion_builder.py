"""
Define subsystem builder for Aviary core propulsion.

Classes
-------
PropulsionBuilderBase : the interface for a propulsion subsystem builder.

CorePropulsionBuilder : the interface for Aviary's core propulsion subsystem builder
"""
from aviary.interface.utils.markdown_utils import write_markdown_variable_table
from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.subsystems.propulsion.propulsion_premission import PropulsionPreMission
from aviary.subsystems.propulsion.propulsion_mission import PropulsionMission
from aviary.variable_info.variables import Aircraft

_default_name = 'propulsion'


class PropulsionBuilderBase(SubsystemBuilderBase):
    def __init__(self, name=None, meta_data=None):
        if name is None:
            name = _default_name

        super().__init__(name=name, meta_data=meta_data)

    def mission_inputs(self, **kwargs):
        return ['*']

    def mission_outputs(self, **kwargs):
        return ['*']


class CorePropulsionBuilder(PropulsionBuilderBase):
    # code_origin is not necessary for this subsystem, catch with kwargs and ignore
    def __init__(self, name=None, meta_data=None, **kwargs):
        if name is None:
            name = 'core_propulsion'

        super().__init__(name=name, meta_data=meta_data)

    def build_pre_mission(self, aviary_inputs):
        return PropulsionPreMission(aviary_options=aviary_inputs)

    def build_mission(self, num_nodes, aviary_inputs, **kwargs):
        return PropulsionMission(num_nodes=num_nodes, aviary_options=aviary_inputs)

    def report(self, prob, reports_folder, **kwargs):
        """
        Generate the report for Aviary core propulsion analysis

        Parameters
        ----------
        prob : AviaryProblem
            The AviaryProblem that will be used to generate the report
        reports_folder : Path
            Location of the subsystems_report folder this report will be placed in
        """
        filename = self.name + '.md'
        filepath = reports_folder / filename

        propulsion_outputs = [Aircraft.Propulsion.TOTAL_NUM_ENGINES,
                              Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST]

        engine_outputs = [Aircraft.Engine.NUM_ENGINES,
                          Aircraft.Engine.SCALE_FACTOR,
                          Aircraft.Engine.SCALED_SLS_THRUST]

        with open(filepath, mode='w') as f:
            f.write('# PROPULSION')
            write_markdown_variable_table(f, prob, propulsion_outputs, self.meta_data)
            f.write('\n## ENGINES')
            for engine in prob.aviary_inputs.get_val('engine_models'):
                f.write(f'\n### {engine.name}')
                write_markdown_variable_table(f, engine, engine_outputs, self.meta_data)
