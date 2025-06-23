"""
Define subsystem builder for Aviary core propulsion.

Classes
-------
PropulsionBuilderBase : the interface for a propulsion subsystem builder.

CorePropulsionBuilder : the interface for Aviary's core propulsion subsystem builder
"""

import numpy as np

from aviary.interface.utils.markdown_utils import write_markdown_variable_table
from aviary.subsystems.propulsion.engine_model import EngineModel
from aviary.subsystems.propulsion.propulsion_mission import PropulsionMission
from aviary.subsystems.propulsion.propulsion_premission import PropulsionPreMission
from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase

# NOTE These are currently needed to get around variable hierarchy being class-based.
#      Ideally, an alternate solution to loop through the hierarchy will be created and
#      these can be replaced.
from aviary.utils.preprocessors import _get_engine_variables
from aviary.variable_info.variables import Aircraft

_default_name = 'propulsion'


class PropulsionBuilderBase(SubsystemBuilderBase):
    """
    Base class for propulsion builder.

    Note
    ----
    unlike the other subsystem builders, it is not recommended to create additional
    propulsion subsystems, as propulsion is intended to be an agnostic carrier of
    all propulsion-related subsystem builders in the form of EngineModels.

    Methods
    -------
    mission_inputs()
        class method to return mission inputs.
    mission_outputs()
        class method to return mission outputs.
    """

    def __init__(self, name=None, meta_data=None):
        if name is None:
            name = _default_name

        super().__init__(name=name, meta_data=meta_data)

    def mission_inputs(self, **kwargs):
        return ['*']

    def mission_outputs(self, **kwargs):
        return ['*']


class CorePropulsionBuilder(PropulsionBuilderBase):
    """
    Core propulsion builder.

    Methods
    -------
    build_pre_mission(self, aviary_inputs) -> openmdao.core.System:
        Build pre-mission.
    build_mission(self, num_nodes, aviary_inputs, **kwargs) -> openmdao.core.System:
        Build pre-mission.
    get_states(self) -> dict:
        Call get_states() on all engine models and return combined result.
    get_parameters(self, aviary_inputs=None, phase_info=None) -> dict:
        Set expected shape of all variables that need to be vectorized for multiple engine types.
    get_controls(self, phase_name=None) -> dict:
        Call get_controls() on all engine models and return combined result.
    get_linked_variables(self) -> dict:
        Call get_linked_variables() on all engine models and return combined result.
    get_pre_mission_bus_variables(self) -> dict
        Call get_linked_variables() on all engine models and return combined result.
    define_order(self) -> list:
        Call define_order() on all engine models and return combined result.
    get_design_vars(self) -> dict:
        Call get_design_vars() on all engine models and return combined result.
    get_initial_guesses(self) -> dict:
        Call get_initial_guesses() on all engine models and return combined result.
    get_mass_names(self) -> list:
        Call get_mass_names() on all engine models and return combined result.
    preprocess_inputs(self) -> aviary_inputs:
        Call get_mass_names() on all engine models and return combined result.
    get_outputs(self) -> list:
        Call get_outputs() on all engine models and return combined result.
    report(self):
        Generate the report for Aviary core propulsion analysis.
    """

    def __init__(self, name=None, meta_data=None, engine_models=None, **kwargs):
        if name is None:
            name = 'core_propulsion'

        super().__init__(name=name, meta_data=meta_data)

        if not isinstance(engine_models, (list, np.ndarray)):
            engine_models = [engine_models]

        for engine in engine_models:
            if not isinstance(engine, EngineModel):
                raise UserWarning(
                    'Engine provided to propulsion builder is not an EngineModel object'
                )

        self.engine_models = engine_models

    def build_pre_mission(self, aviary_inputs, **kwargs):
        return PropulsionPreMission(
            aviary_options=aviary_inputs, engine_models=self.engine_models, engine_options=kwargs
        )

    def build_mission(self, num_nodes, aviary_inputs, **kwargs):
        return PropulsionMission(
            num_nodes=num_nodes,
            aviary_options=aviary_inputs,
            engine_models=self.engine_models,
            engine_options=kwargs,
        )

    # NOTE no unittests!
    def get_states(self):
        """Call get_states() on all engine models and return combined result."""
        states = {}
        for engine in self.engine_models:
            engine_states = engine.get_states()
            states.update(engine_states)

        return states

    def get_controls(self, phase_name=None):
        """Call get_controls() on all engine models and return combined result."""
        controls = {}
        for engine in self.engine_models:
            engine_controls = engine.get_controls(phase_name=phase_name)
            controls.update(engine_controls)

        return controls

    # NOTE no unittests!
    def get_parameters(self, aviary_inputs=None, phase_info=None):
        """
        Set expected shape of all variables that need to be vectorized for multiple
        engine types.
        """
        num_engine_type = len(aviary_inputs.get_val(Aircraft.Engine.NUM_ENGINES))
        params = {}

        # collect all the parameters for engines
        for engine in self.engine_models:
            engine_params = engine.get_parameters()
            # for param in engine_params:
            #     # For any parameters that need to be vectorized for multiple engines,
            #     # apply correct shape
            #     if param in params:
            #         try:
            #             shape_old = params[param]['shape'][0]
            #         except KeyError:
            #             # If shape is not defined yet, this is the first time there is
            #             # a duplicate
            #             shape_old = 1
            #         engine_params[param]['shape'] = (shape_old + 1,)

            params.update(engine_params)

        # for any parameters that need to be vectorized for multiple engines, apply
        # correct shape
        engine_vars = [var for var in _get_engine_variables()]
        for var in params:
            if var in engine_vars:
                # TODO shape for variables that are supposed to be vectors, like wing
                #      engine locations
                params[var]['shape'] = (num_engine_type,)
                params[var]['static_target'] = True

        return params

    # NOTE no unittests!
    def get_constraints(self):
        """Call get_constraints() on all engine models and return combined result."""
        constraints = {}
        for engine in self.engine_models:
            engine_constraints = engine.get_constraints()
            constraints.update(engine_constraints)

        return constraints

    # NOTE no unittests!
    def get_linked_variables(self):
        """Call get_linked_variables() on all engine models and return combined result."""
        linked_vars = {}
        for engine in self.engine_models:
            engine_linked_vars = engine.get_linked_variables()
            linked_vars.update(engine_linked_vars)

        return linked_vars

    def get_pre_mission_bus_variables(self, aviary_inputs=None):
        """Call get_linked_variables() on all engine models and return combined result."""
        bus_vars = {}
        for engine in self.engine_models:
            engine_bus_vars = engine.get_pre_mission_bus_variables(aviary_inputs)
            bus_vars.update(engine_bus_vars)

        # append propulsion group name to all engine-level bus variables
        # engine models only need to use variable paths starting at that engine group
        complete_bus_vars = {}
        for var in bus_vars:
            info = bus_vars[var]
            complete_bus_vars[self.name + '.' + var] = info

        return complete_bus_vars

    # NOTE no unittests!
    def define_order(self):
        """Call define_order() on all engine models and return combined result."""
        subsys_order = []
        for engine in self.engine_models:
            engine_subsys_order = engine.define_order()
            subsys_order.append(engine_subsys_order)

        return subsys_order

    # NOTE no unittests!
    def get_design_vars(self):
        """Call get_design_vars() on all engine models and return combined result."""
        design_vars = {}
        for engine in self.engine_models:
            engine_design_vars = engine.get_design_vars()
            design_vars.update(engine_design_vars)

        return design_vars

    def get_initial_guesses(self):
        """Call get_initial_guesses() on all engine models and return combined result."""
        initial_guesses = {}
        for engine in self.engine_models:
            engine_initial_guesses = engine.get_initial_guesses()
            initial_guesses.update(engine_initial_guesses)

        return initial_guesses

    # NOTE no unittests!
    def get_mass_names(self):
        """Call get_mass_names() on all engine models and return combined result."""
        mass_names = {}
        for engine in self.engine_models:
            engine_mass_names = engine.get_mass_names()
            mass_names.update(engine_mass_names)

        return mass_names

    # NOTE no unittests!
    def preprocess_inputs(self):
        """Call get_mass_names() on all engine models and return combined result."""
        mass_names = {}
        for engine in self.engine_models:
            engine_mass_names = engine.get_mass_names()
            mass_names.update(engine_mass_names)

        return mass_names

    # NOTE no unittests!
    def get_outputs(self):
        """Call get_outputs() on all engine models and return combined result."""
        outputs = []
        for engine in self.engine_models:
            engine_outputs = engine.get_outputs()
            outputs.append(engine_outputs)

        return outputs

    def report(self, prob, reports_folder, **kwargs):
        """
        Generate the report for Aviary core propulsion analysis.

        Parameters
        ----------
        prob : AviaryProblem
            The AviaryProblem that will be used to generate the report
        reports_folder : Path
            Location of the subsystems_report folder this report will be placed in
        """
        filename = self.name + '.md'
        filepath = reports_folder / filename

        propulsion_outputs = [
            Aircraft.Propulsion.TOTAL_NUM_ENGINES,
            Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST,
        ]

        with open(filepath, mode='w') as f:
            f.write('# Propulsion')
            write_markdown_variable_table(f, prob, propulsion_outputs, self.meta_data)
            f.write('\n## Engines')

        # each engine can append to this file
        kwargs['meta_data'] = self.meta_data
        for idx, engine in enumerate(self.engine_models):
            kwargs['engine_idx'] = idx
            engine.report(prob, filepath, **kwargs)
