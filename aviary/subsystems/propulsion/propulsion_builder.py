"""
Define subsystem builder for Aviary core propulsion.

Classes
-------
PropulsionBuilderBase : the interface for a propulsion subsystem builder.

CorePropulsionBuilder : the interface for Aviary's core propulsion subsystem builder
"""

import numpy as np

from aviary.interface.utils.markdown_utils import write_markdown_variable_table

from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.subsystems.propulsion.propulsion_premission import PropulsionPreMission
from aviary.subsystems.propulsion.propulsion_mission import PropulsionMission
from aviary.subsystems.propulsion.engine_model import EngineModel

from aviary.variable_info.variables import Aircraft

# NOTE These are currently needed to get around variable hierarchy being class-based.
#      Ideally, an alternate solution to loop through the hierarchy will be created and
#      these can be replaced.
from aviary.utils.preprocessors import _get_engine_variables

_default_name = 'propulsion'


# NOTE unlike the other subsystem builders, it is not reccomended to create additional
#      propulsion subsystems, as propulsion is intended to be an agnostic carrier of
#      all propulsion-related subsystem builders.
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

        try:
            engine_models = kwargs['engine_models']
        except KeyError:
            engine_models = None
        else:
            if not isinstance(engine_models, (list, np.ndarray)):
                engine_models = [engine_models]

            for engine in engine_models:
                if not isinstance(engine, EngineModel):
                    raise UserWarning('Engine provided to propulsion builder is not an '
                                      'EngineModel object')

        self.engine_models = engine_models

    def build_pre_mission(self, aviary_inputs):
        return PropulsionPreMission(aviary_options=aviary_inputs,
                                    engine_models=self.engine_models)

    def build_mission(self, num_nodes, aviary_inputs, **kwargs):
        return PropulsionMission(num_nodes=num_nodes, aviary_options=aviary_inputs,
                                 engine_models=self.engine_models)

    # NOTE untested!
    def get_states(self):
        """
        Call get_states() on all engine models and return combined result.
        """
        states = {}
        for engine in self.engine_models:
            engine_states = engine.get_states()
            states.update(engine_states)

        return states

    # NOTE untested!
    def get_controls(self):
        """
        Call get_controls() on all engine models and return combined result.
        """
        controls = {}
        for engine in self.engine_models:
            engine_controls = engine.get_controls()
            controls.update(engine_controls)

        return controls

    # TODO add parameters defined by individual engines, update to correct shape if necessary
    def get_parameters(self, aviary_inputs=None, phase_info=None):
        """
        Set expected shape of all variables that need to be vectorized for multiple
        engine types.
        """
        engine_count = len(aviary_inputs.get_val(Aircraft.Engine.NUM_ENGINES))
        params = {}

        # add all variables from Engine & Nacelle to params
        # TODO this assumes that no new categories are added for custom engine models
        for var in _get_engine_variables():
            if var in aviary_inputs:
                # TODO engine_wing_location
                params[var] = {'shape': (engine_count, ), 'static_target': True}

        # params = {}  # For now
        # params[Aircraft.Engine.SCALE_FACTOR] = {'shape': (engine_count, ),
        #                                         'static_target': True}
        return params

    # NOTE untested!
    def get_constraints(self):
        """
        Call get_constraints() on all engine models and return combined result.
        """
        constraints = {}
        for engine in self.engine_models:
            engine_constraints = engine.get_constraints()
            constraints.update(engine_constraints)

        return constraints

    # NOTE untested!
    def get_linked_variables(self):
        """
        Call get_linked_variables() on all engine models and return combined result.
        """
        linked_vars = {}
        for engine in self.engine_models:
            engine_linked_vars = engine.get_linked_variables()
            linked_vars.update(engine_linked_vars)

        return linked_vars

    # NOTE untested!
    def get_bus_variables(self):
        """
        Call get_linked_variables() on all engine models and return combined result.
        """
        linked_vars = {}
        for engine in self.engine_models:
            engine_linked_vars = engine.get_linked_variables()
            linked_vars.update(engine_linked_vars)

        return linked_vars

    # NOTE untested!
    def define_order(self):
        """
        Call define_order() on all engine models and return combined result.
        """
        subsys_order = []
        for engine in self.engine_models:
            engine_subsys_order = engine.define_order()
            subsys_order.append(engine_subsys_order)

        return subsys_order

    # NOTE untested!
    def get_design_vars(self):
        """
        Call get_design_vars() on all engine models and return combined result.
        """
        design_vars = {}
        for engine in self.engine_models:
            engine_design_vars = engine.get_design_vars()
            design_vars.update(engine_design_vars)

        return design_vars

    # NOTE untested!
    def get_initial_guesses(self):
        """
        Call get_initial_guesses() on all engine models and return combined result.
        """
        initial_guesses = {}
        for engine in self.engine_models:
            engine_initial_guesses = engine.get_initial_guesses()
            initial_guesses.update(engine_initial_guesses)

        return initial_guesses

    # NOTE untested!
    def get_mass_names(self):
        """
        Call get_mass_names() on all engine models and return combined result.
        """
        mass_names = {}
        for engine in self.engine_models:
            engine_mass_names = engine.get_mass_names()
            mass_names.update(engine_mass_names)

        return mass_names

    # NOTE untested!
    def preprocess_inputs(self):
        """
        Call get_mass_names() on all engine models and return combined result.
        """
        mass_names = {}
        for engine in self.engine_models:
            engine_mass_names = engine.get_mass_names()
            mass_names.update(engine_mass_names)

        return mass_names

    # NOTE untested!
    def get_outputs(self):
        """
        Call get_outputs() on all engine models and return combined result.
        """
        outputs = []
        for engine in self.engine_models:
            engine_outputs = engine.get_outputs()
            outputs.append(engine_outputs)

        return outputs

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

        with open(filepath, mode='w') as f:
            f.write('# Propulsion')
            write_markdown_variable_table(f, prob, propulsion_outputs, self.meta_data)
            f.write('\n## Engines')

        # each engine can append to this file
        kwargs['meta_data'] = self.meta_data
        for engine in prob.aviary_inputs.get_val('engine_models'):
            engine.report(prob, filepath, **kwargs)
