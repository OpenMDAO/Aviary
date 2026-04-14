import warnings

import numpy as np
import openmdao.api as om

from aviary.subsystems.propulsion.engine_deck import EngineDeck
from aviary.subsystems.propulsion.engine_model import EngineModel
from aviary.subsystems.propulsion.gearbox.gearbox_builder import GearboxBuilder
from aviary.subsystems.propulsion.propeller.propeller_builder import PropellerBuilder
from aviary.subsystems.propulsion.utils import EngineModelVariables, build_engine_deck
from aviary.subsystems.subsystem_builder import SubsystemBuilder
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import Verbosity
from aviary.variable_info.variables import Aircraft, Dynamic, Settings


class TurbopropModel(EngineModel):
    """
    EngineModel that combines a model for shaft power generation (default is EngineDeck), a gearbox
    model (default is simple gearbox), and a model for propeller performance (default is Hamilton
    Standard).

    Attributes
    ----------
    name : str ('engine')
        Object label.
    options : AviaryValues (<empty>)
        Inputs and options related to engine model.
    shaft_power_model : SubsystemBuilder (<empty>)
        Subsystem builder for the shaft power generating component. If None, an EngineDeck built
        using provided options is used.
    gearbox_model : SubsystemBuilder (<empty>)
        Subsystem builder used for the gearbox. If None, the simple gearbox model is used.
    propeller_model : SubsystemBuilder (<empty>)
        Subsystem builder for the propeller. If None, the Hamilton Standard methodology is used to
        model the propeller.

    Methods
    -------
    - build_pre_mission
    - build_mission
    - build_post_mission
    - needs_mission_solver
    - get_states
    - get_controls
    - get_parameters
    - get_constraints
    - get_bus_variables
    - get_pre_mission_bus_variables
    - get_linked_variables
    - get_design_vars
    - get_initial_guesses
    - get_mass_names
    - get_val
    - get_item
    - set_val
    - update
    """

    def __init__(
        self,
        name='turboprop_model',
        options: AviaryValues = None,
        shaft_power_model: SubsystemBuilder = None,
        propeller_model: SubsystemBuilder = None,
        gearbox_model: SubsystemBuilder = None,
    ):
        # also calls _preprocess_inputs() as part of EngineModel __init__
        super().__init__(name, options)

        self.shaft_power_model = shaft_power_model
        self.propeller_model = propeller_model
        self.gearbox_model = gearbox_model

        # Initialize turboshaft engine deck. New required variable set w/o thrust
        if shaft_power_model is None:
            self.shaft_power_model = build_engine_deck(
                name='engine_deck',
                options=options,
                required_variables={
                    EngineModelVariables.ALTITUDE,
                    EngineModelVariables.MACH,
                    EngineModelVariables.THROTTLE,
                },
            )

        # TODO No reason gearbox model needs to be required. All connections can be handled in
        #      configure - need to figure out when user wants gearbox without having to directly
        #      pass builder
        if gearbox_model is None:
            # TODO where can we bring in include_constraints? kwargs in init is an option, but that
            #      still requires the L2 interface
            self.gearbox_model = GearboxBuilder(name='gearbox', include_constraints=True)

        if propeller_model is None:
            self.propeller_model = PropellerBuilder(name='propeller')

    def needs_mission_solver(self, aviary_inputs, subsystem_options):
        if self.shaft_power_model is not None:
            shp_solver = self.shaft_power_model.needs_mission_solver(
                aviary_inputs=aviary_inputs,
                subsystem_options=subsystem_options,
            )
        else:
            shp_solver = False
        if self.gearbox_model is not None:
            gearbox_solver = self.gearbox_model.needs_mission_solver(
                aviary_inputs=aviary_inputs,
                subsystem_options=subsystem_options,
            )
        else:
            gearbox_solver = False
        if self.propeller_model is not None:
            prop_solver = self.propeller_model.needs_mission_solver(
                aviary_inputs=aviary_inputs,
                subsystem_options=subsystem_options,
            )
        else:
            prop_solver = False
        mission_solver = np.any([shp_solver, gearbox_solver, prop_solver])
        return mission_solver

    # BUG if using multiple custom subsystems that happen to share a kwarg but need different values,
    #     this breaks - look into "nested" kwargs with separate dict per turboprop subsystem?
    def build_pre_mission(self, aviary_inputs, subsystem_options) -> om.Group:
        shp_model = self.shaft_power_model
        propeller_model = self.propeller_model
        gearbox_model = self.gearbox_model
        turboprop_group = om.Group()

        # TODO engine scaling for turboshafts requires EngineSizing to be refactored to accept
        #      target scaling variable as an option, skipping for now
        if not isinstance(shp_model, EngineDeck):
            shp_model_pre_mission = shp_model.build_pre_mission(
                self.options, subsystem_options=subsystem_options
            )
            if shp_model_pre_mission is not None:
                turboprop_group.add_subsystem(
                    shp_model.name, subsys=shp_model_pre_mission, promotes=['*']
                )

        gearbox_model_pre_mission = gearbox_model.build_pre_mission(
            self.options, subsystem_options=subsystem_options
        )
        if gearbox_model_pre_mission is not None:
            turboprop_group.add_subsystem(
                gearbox_model.name,
                subsys=gearbox_model_pre_mission,
                promotes=['*'],
            )

        propeller_model_pre_mission = propeller_model.build_pre_mission(
            self.options, subsystem_options=subsystem_options
        )
        if propeller_model_pre_mission is not None:
            turboprop_group.add_subsystem(
                propeller_model.name,
                subsys=propeller_model_pre_mission,
                promotes=['*'],
            )

        return turboprop_group

    def build_mission(self, num_nodes, aviary_inputs, user_options, subsystem_options):
        turboprop_group = TurbopropMission(
            num_nodes=num_nodes,
            shaft_power_model=self.shaft_power_model,
            propeller_model=self.propeller_model,
            gearbox_model=self.gearbox_model,
            aviary_inputs=self.options,
            kwargs=subsystem_options,
        )

        return turboprop_group

    def mission_inputs(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        # NOTE what may be an input/output for an individual subsystem may not be an overall
        # input/output for the TurbopropModel as a whole, commenting this out for now
        # inputs = []
        # if self.shaft_power_model is not None:
        #     if self.shaft_power_model.name in kwargs:
        #         subsys_args = kwargs[self.shaft_power_model.name]
        #     else:
        #         subsys_args = {}
        #     inputs += self.shaft_power_model.mission_inputs(**subsys_args)
        # if self.gearbox_model is not None:
        #     if self.gearbox_model.name in kwargs:
        #         subsys_args = kwargs[self.gearbox_model.name]
        #     else:
        #         subsys_args = {}
        #     inputs += self.gearbox_model.mission_inputs(**subsys_args)
        # if self.propeller_model is not None:
        #     if self.propeller_model.name in kwargs:
        #         subsys_args = kwargs[self.propeller_model.name]
        #     else:
        #         subsys_args = {}
        #     inputs += self.propeller_model.mission_inputs(**subsys_args)
        # return list(set(inputs))
        return []

    def mission_outputs(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        # NOTE what may be an input/output for an individual subsystem may not be an overall
        # input/output for the TurbopropModel as a whole, commenting this out for now
        # outputs = []
        # if self.shaft_power_model is not None:
        #     if self.shaft_power_model.name in kwargs:
        #         subsys_args = kwargs[self.shaft_power_model.name]
        #     else:
        #         subsys_args = {}
        #     outputs += self.shaft_power_model.mission_outputs(**subsys_args)
        # if self.gearbox_model is not None:
        #     if self.gearbox_model.name in kwargs:
        #         subsys_args = kwargs[self.gearbox_model.name]
        #     else:
        #         subsys_args = {}
        #     outputs += self.gearbox_model.mission_outputs(**subsys_args)
        # if self.propeller_model is not None:
        #     if self.propeller_model.name in kwargs:
        #         subsys_args = kwargs[self.propeller_model.name]
        #     else:
        #         subsys_args = {}
        #     outputs += self.propeller_model.mission_outputs(**subsys_args)
        # return list(set(outputs))
        return []

    def build_post_mission(
        self,
        aviary_inputs=None,
        mission_info=None,
        subsystem_options=None,
        phase_mission_bus_lengths=None,
    ):
        shp_model = self.shaft_power_model
        gearbox_model = self.gearbox_model
        propeller_model = self.propeller_model
        turboprop_group = om.Group()

        shp_model_post_mission = shp_model.build_post_mission(
            aviary_inputs=aviary_inputs,
            mission_info=mission_info,
            subsystem_options=subsystem_options,
            phase_mission_bus_lengths=phase_mission_bus_lengths,
        )

        if shp_model_post_mission is not None:
            turboprop_group.add_subsystem(
                shp_model.name,
                subsys=shp_model_post_mission,
            )

        gearbox_model_post_mission = gearbox_model.build_post_mission(
            aviary_inputs=aviary_inputs,
            mission_info=mission_info,
            subsystem_options=subsystem_options,
            phase_mission_bus_lengths=phase_mission_bus_lengths,
        )

        if gearbox_model_post_mission is not None:
            turboprop_group.add_subsystem(
                gearbox_model.name,
                subsys=gearbox_model_post_mission,
            )

        propeller_model_post_mission = propeller_model.build_post_mission(
            aviary_inputs=aviary_inputs,
            mission_info=mission_info,
            subsystem_options=subsystem_options,
            phase_mission_bus_lengths=phase_mission_bus_lengths,
        )

        if propeller_model_post_mission is not None:
            turboprop_group.add_subsystem(
                propeller_model.name,
                subsys=propeller_model_post_mission,
            )

        return turboprop_group

    def get_states(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        states = super().get_states(
            aviary_inputs=aviary_inputs,
            user_options=user_options,
            subsystem_options=subsystem_options,
        )

        if self.shaft_power_model is not None:
            extra_states = self.shaft_power_model.get_states(
                aviary_inputs=aviary_inputs,
                user_options=user_options,
                subsystem_options=subsystem_options,
            )
            states.update(extra_states)

        if self.gearbox_model is not None:
            extra_states = self.gearbox_model.get_states(
                aviary_inputs=aviary_inputs,
                user_options=user_options,
                subsystem_options=subsystem_options,
            )
            states.update(extra_states)

        if self.propeller_model is not None:
            extra_states = self.propeller_model.get_states(
                aviary_inputs=aviary_inputs,
                user_options=user_options,
                subsystem_options=subsystem_options,
            )
            states.update(extra_states)

        return states

    def get_controls(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        controls = super().get_controls(
            aviary_inputs=aviary_inputs,
            user_options=user_options,
            subsystem_options=subsystem_options,
        )

        if self.shaft_power_model is not None:
            extra_controls = self.shaft_power_model.get_controls(
                aviary_inputs=aviary_inputs,
                user_options=user_options,
                subsystem_options=subsystem_options,
            )
            controls.update(extra_controls)

        if self.gearbox_model is not None:
            extra_controls = self.gearbox_model.get_controls(
                aviary_inputs=aviary_inputs,
                user_options=user_options,
                subsystem_options=subsystem_options,
            )
            controls.update(extra_controls)

        if self.propeller_model is not None:
            extra_controls = self.propeller_model.get_controls(
                aviary_inputs=aviary_inputs,
                user_options=user_options,
                subsystem_options=subsystem_options,
            )
            controls.update(extra_controls)

        # RPM control needs special handling, which is done automatically in configure step of
        # Mission group
        if Dynamic.Vehicle.Propulsion.RPM in controls:
            controls[f'{Dynamic.Vehicle.Propulsion.RPM}_control'] = controls.pop(
                Dynamic.Vehicle.Propulsion.RPM
            )

        return controls

    def get_constraints(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        constraints = super().get_constraints(
            aviary_inputs=aviary_inputs,
            user_options=user_options,
            subsystem_options=subsystem_options,
        )  # calls from EngineModel

        if self.shaft_power_model is not None:
            extra_constraints = self.shaft_power_model.get_constraints(
                aviary_inputs=aviary_inputs,
                user_options=user_options,
                subsystem_options=subsystem_options,
            )
            constraints.update(extra_constraints)

        if self.gearbox_model is not None:
            extra_constraints = self.gearbox_model.get_constraints(
                aviary_inputs=aviary_inputs,
                user_options=user_options,
                subsystem_options=subsystem_options,
            )
            constraints.update(extra_constraints)

        if self.propeller_model is not None:
            extra_constraints = self.propeller_model.get_constraints(
                aviary_inputs=aviary_inputs,
                user_options=user_options,
                subsystem_options=subsystem_options,
            )
            constraints.update(extra_constraints)

        return constraints

    def get_parameters(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        params = super().get_parameters(
            aviary_inputs=aviary_inputs,
            user_options=user_options,
            subsystem_options=subsystem_options,
        )  # calls from EngineModel

        if self.shaft_power_model is not None:
            extra_params = self.shaft_power_model.get_parameters(
                aviary_inputs=aviary_inputs,
                user_options=user_options,
                subsystem_options=subsystem_options,
            )
            params.update(extra_params)

        if self.gearbox_model is not None:
            extra_params = self.gearbox_model.get_parameters(
                aviary_inputs=aviary_inputs,
                user_options=user_options,
                subsystem_options=subsystem_options,
            )
            params.update(extra_params)

        if self.propeller_model is not None:
            extra_params = self.propeller_model.get_parameters(
                aviary_inputs=aviary_inputs,
                user_options=user_options,
                subsystem_options=subsystem_options,
            )
            params.update(extra_params)

        return params

    def get_linked_variables(self, aviary_inputs=None):
        linked_vars = super().get_linked_variables()  # calls from EngineModel
        if self.shaft_power_model is not None:
            linked_vars += self.shaft_power_model.get_linked_variables(aviary_inputs=aviary_inputs)
        if self.gearbox_model is not None:
            linked_vars += self.gearbox_model.get_linked_variables(aviary_inputs=aviary_inputs)
        if self.propeller_model is not None:
            linked_vars += self.propeller_model.get_linked_variables(aviary_inputs=aviary_inputs)

        return linked_vars

    def get_bus_variables(self, aviary_inputs=None):
        busvars = super().get_bus_variables()  # calls from EngineModel
        if self.shaft_power_model is not None:
            busvars.update(self.shaft_power_model.get_bus_variables(aviary_inputs=aviary_inputs))
        if self.gearbox_model is not None:
            busvars.update(self.gearbox_model.get_bus_variables(aviary_inputs=aviary_inputs))
        if self.propeller_model is not None:
            busvars.update(self.propeller_model.get_bus_variables(aviary_inputs=aviary_inputs))
        return busvars

    def get_pre_mission_bus_variables(self, aviary_inputs=None, mission_info=None):
        bus_vars = super().get_pre_mission_bus_variables(
            aviary_inputs=aviary_inputs,
            mission_info=mission_info,
        )  # calls from EngineModel

        if self.shaft_power_model is not None:
            extra_bus_vars = self.shaft_power_model.get_pre_mission_bus_variables(
                aviary_inputs=aviary_inputs, mission_info=mission_info
            )
            bus_vars.update(extra_bus_vars)

        if self.gearbox_model is not None:
            extra_bus_vars = self.gearbox_model.get_pre_mission_bus_variables(
                aviary_inputs=aviary_inputs, mission_info=mission_info
            )
            bus_vars.update(extra_bus_vars)

        if self.propeller_model is not None:
            extra_bus_vars = self.propeller_model.get_pre_mission_bus_variables(
                aviary_inputs=aviary_inputs, mission_info=mission_info
            )
            bus_vars.update(extra_bus_vars)

        return bus_vars

    def get_design_vars(self, aviary_inputs=None):
        desvars = super().get_design_vars()  # calls from EngineModel
        if self.shaft_power_model is not None:
            desvars.update(self.shaft_power_model.get_design_vars(aviary_inputs=aviary_inputs))
        if self.gearbox_model is not None:
            desvars.update(self.gearbox_model.get_design_vars(aviary_inputs=aviary_inputs))
        if self.propeller_model is not None:
            desvars.update(self.propeller_model.get_design_vars(aviary_inputs=aviary_inputs))
        return desvars

    def get_initial_guesses(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        guesses = super().get_initial_guesses(
            aviary_inputs=aviary_inputs,
            user_options=user_options,
            subsystem_options=subsystem_options,
        )

        if self.shaft_power_model is not None:
            extra_guesses = self.shaft_power_model.get_initial_guesses(
                aviary_inputs=aviary_inputs,
                user_options=user_options,
                subsystem_options=subsystem_options,
            )
            guesses.update(extra_guesses)

        if self.gearbox_model is not None:
            extra_guesses = self.gearbox_model.get_initial_guesses(
                aviary_inputs=aviary_inputs,
                user_options=user_options,
                subsystem_options=subsystem_options,
            )
            guesses.update(extra_guesses)

        if self.propeller_model is not None:
            extra_guesses = self.propeller_model.get_initial_guesses(
                aviary_inputs=aviary_inputs,
                user_options=user_options,
                subsystem_options=subsystem_options,
            )
            guesses.update(extra_guesses)

        return guesses

    def get_mass_names(self, aviary_inputs=None):
        mass_names = super().get_mass_names()  # calls from EngineModel
        if self.shaft_power_model is not None:
            mass_names += self.shaft_power_model.get_mass_names(aviary_inputs=aviary_inputs)
        if self.gearbox_model is not None:
            mass_names += self.gearbox_model.get_mass_names(aviary_inputs=aviary_inputs)
        if self.propeller_model is not None:
            mass_names += self.propeller_model.get_mass_names(aviary_inputs=aviary_inputs)
        return mass_names

    def preprocess_inputs(self, aviary_inputs=None):
        new_inputs = super().preprocess_inputs(aviary_inputs)
        if self.shaft_power_model is not None:
            new_inputs.update(self.shaft_power_model.preprocess_inputs(aviary_inputs=new_inputs))
        if self.gearbox_model is not None:
            new_inputs.update(self.gearbox_model.get_mass_names(aviary_inputs=new_inputs))
        if self.propeller_model is not None:
            new_inputs.update(self.propeller_model.get_mass_names(aviary_inputs=new_inputs))
        return new_inputs

    def get_timeseries(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        new_timeseries = super().get_timeseries(
            aviary_inputs, user_options, subsystem_options
        )  # calls from EngineModel
        if self.shaft_power_model is not None:
            new_timeseries += self.shaft_power_model.get_timeseries(
                aviary_inputs, user_options, subsystem_options
            )
        if self.gearbox_model is not None:
            new_timeseries += self.gearbox_model.get_timeseries(
                aviary_inputs, user_options, subsystem_options
            )
        if self.propeller_model is not None:
            new_timeseries += self.propeller_model.get_timeseries(
                aviary_inputs, user_options, subsystem_options
            )
        return new_timeseries

    def get_post_mission_bus_variables(self, aviary_inputs=None, mission_info=None):
        new_bus = super().get_post_mission_bus_variables(
            aviary_inputs, mission_info
        )  # calls from EngineModel
        if self.shaft_power_model is not None:
            new_bus.update(
                self.shaft_power_model.get_post_mission_bus_variables(aviary_inputs, mission_info)
            )
        if self.gearbox_model is not None:
            new_bus.update(
                self.gearbox_model.get_post_mission_bus_variables(aviary_inputs, mission_info)
            )
        if self.propeller_model is not None:
            new_bus.update(
                self.propeller_model.get_post_mission_bus_variables(aviary_inputs, mission_info)
            )
        return new_bus

    def report(self, prob, reports_folder, **kwargs):
        if self.shaft_power_model is not None:
            self.shaft_power_model.report(prob, reports_folder, **kwargs)
        if self.gearbox_model is not None:
            self.gearbox_model.report(prob, reports_folder, **kwargs)
        if self.propeller_model is not None:
            self.propeller_model.report(prob, reports_folder, **kwargs)


class TurbopropMission(om.Group):
    def initialize(self):
        self.options.declare(
            'num_nodes', types=int, desc='Number of nodes to be evaluated in the RHS'
        )
        self.options.declare('shaft_power_model', desc='shaft power generation model')
        self.options.declare('propeller_model', desc='propeller model')
        self.options.declare('gearbox_model', desc='gearbox model')
        self.options.declare('kwargs', desc='kwargs for turboprop mission model')
        self.options.declare('aviary_inputs', desc='aviary inputs for turboprop mission model')

    def setup(self):
        # All promotions for configurable components in this group are handled during configure()

        num_nodes = self.options['num_nodes']
        shp_model = self.options['shaft_power_model']
        propeller_model = self.options['propeller_model']
        gearbox_model = self.options['gearbox_model']
        kwargs = self.options['kwargs']
        aviary_inputs = self.options['aviary_inputs']

        if Aircraft.Engine.FIXED_RPM in aviary_inputs:
            # NOTE this subsystem is a empty component that has fixed RPM added as an output in
            #      configure() if provided in aviary_inputs
            self.add_subsystem('rpm_source', subsys=om.IndepVarComp())
        else:
            # passthrough component so RPM can be used as a control
            self.add_subsystem(
                'rpm_source',
                subsys=om.ExecComp(
                    'RPM=RPM_control',
                    RPM={'val': np.zeros(num_nodes), 'units': 'rpm'},
                    RPM_control={'val': np.zeros(num_nodes), 'units': 'rpm'},
                    has_diag_partials=True,
                ),
            )

        # Shaft Power Model
        try:
            subsystem_options = kwargs[shp_model.name]
        except (AttributeError, KeyError):
            subsystem_options = {}
        shp_model_mission = shp_model.build_mission(
            num_nodes=num_nodes,
            aviary_inputs=aviary_inputs,
            user_options={},
            subsystem_options=subsystem_options,
        )
        if shp_model_mission is not None:
            self.add_subsystem(shp_model.name, subsys=shp_model_mission)

        # Gearbox Model
        try:
            gearbox_kwargs = kwargs[gearbox_model.name]
        except (AttributeError, KeyError):
            gearbox_kwargs = {}
        if gearbox_model is not None:
            gearbox_model_mission = gearbox_model.build_mission(
                num_nodes=num_nodes,
                aviary_inputs=aviary_inputs,
                user_options={},
                subsystem_options=gearbox_kwargs,
            )
            if gearbox_model_mission is not None:
                self.add_subsystem(gearbox_model.name, subsys=gearbox_model_mission)

        # Propeller Model
        try:
            propeller_kwargs = kwargs[propeller_model.name]
        except (AttributeError, KeyError):
            propeller_kwargs = {}

        propeller_model_mission = propeller_model.build_mission(
            num_nodes=num_nodes,
            aviary_inputs=aviary_inputs,
            user_options={},
            subsystem_options=propeller_kwargs,
        )

        if isinstance(propeller_model, PropellerBuilder):
            # use the Hamilton Standard model
            try:
                propeller_kwargs = kwargs['hamilton_standard']
            except KeyError:
                propeller_kwargs = {}

            self.add_subsystem(propeller_model.name, propeller_model_mission)

        else:
            if propeller_model_mission is not None:
                self.add_subsystem(propeller_model.name, subsys=propeller_model_mission)

        thrust_summation = om.ExecComp(
            'turboprop_thrust=turboshaft_thrust+propeller_thrust',
            turboprop_thrust={'val': np.zeros(num_nodes), 'units': 'lbf'},
            turboshaft_thrust={'val': np.zeros(num_nodes), 'units': 'lbf'},
            propeller_thrust={'val': np.zeros(num_nodes), 'units': 'lbf'},
            has_diag_partials=True,
        )

        self.add_subsystem(
            'thrust_summation',
            subsys=thrust_summation,
            # promotes_inputs=['*'],
            promotes_outputs=[('turboprop_thrust', Dynamic.Vehicle.Propulsion.THRUST)],
        )

    def configure(self):
        """
        Correctly connect variables between shaft power model, gearbox, and propeller.

        When an input in a component is present as an output in a upstream component, the two are
        connected rather than promoted. This prevents intermediate values from "leaking" out
        of the model and getting incorrectly connected to outside components. All other inputs
        and outputs are promoted.

        Set up fixed RPM value if requested by user, which overrides any RPM defined by shaft power
        model
        """
        has_gearbox = self.options['gearbox_model'] is not None
        num_nodes = self.options['num_nodes']
        aviary_inputs = self.options['aviary_inputs']

        # TODO this list shouldn't be hardcoded - it should mirror propulsion_mission list
        # Don't promote inputs that are in this list - shaft power should be an output of this
        # system, also having it as an input causes feedback loop problem at the propulsion level
        skipped_inputs = [
            Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN,
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE,
            Dynamic.Vehicle.Propulsion.NOX_RATE,
            Dynamic.Vehicle.Propulsion.SHAFT_POWER,
            Dynamic.Vehicle.Propulsion.SHAFT_POWER_MAX,
            Dynamic.Vehicle.Propulsion.TEMPERATURE_T4,
            Dynamic.Vehicle.Propulsion.THRUST,
            Dynamic.Vehicle.Propulsion.THRUST_MAX,
        ]

        # Build lists of inputs/outputs for each component as needed:
        # "_input_list" or "_output_list" are all variables that still need to be connected or
        # promoted. This list is pared down as each variable is handled.

        # "_inputs" or "_outputs" is a list that tracks all the promotions needed for a given
        # component, which is done at the very end.

        shp_model = self._get_subsystem(self.options['shaft_power_model'].name)
        shp_input_dict = shp_model.list_inputs(
            return_format='dict', units=True, out_stream=None, all_procs=True
        )
        shp_input_list = list(
            set(
                shp_input_dict[key]['prom_name']
                for key in shp_input_dict
                if '.' not in shp_input_dict[key]['prom_name']
            )
        )
        shp_output_dict = shp_model.list_outputs(
            return_format='dict', units=True, out_stream=None, all_procs=True
        )
        shp_output_list = list(
            set(
                shp_output_dict[key]['prom_name']
                for key in shp_output_dict
                if '.' not in shp_output_dict[key]['prom_name']
            )
        )

        shp_inputs = []
        shp_outputs = []

        if has_gearbox:
            gearbox_model = self._get_subsystem(self.options['gearbox_model'].name)
            gearbox_input_dict = gearbox_model.list_inputs(
                return_format='dict', units=True, out_stream=None, all_procs=True
            )
            # Filter out variables with '_out' from input promotion list. This is necessary because
            # Aviary gearbox uses things like shp_out as an input internally, but for the gearbox
            # subsystem as a whole, it is an output.
            gearbox_input_list = list(
                set(
                    gearbox_input_dict[key]['prom_name']
                    for key in gearbox_input_dict
                    if '.' not in gearbox_input_dict[key]['prom_name']
                    and '_out' not in gearbox_input_dict[key]['prom_name']
                )
            )

            gearbox_output_dict = gearbox_model.list_outputs(
                return_format='dict', units=True, out_stream=None, all_procs=True
            )
            gearbox_output_list = list(
                set(
                    gearbox_output_dict[key]['prom_name']
                    for key in gearbox_output_dict
                    if '.' not in gearbox_output_dict[key]['prom_name']
                )
            )

        gearbox_inputs = []
        gearbox_outputs = []

        propeller_model_name = self.options['propeller_model'].name
        propeller_model = self._get_subsystem(propeller_model_name)
        propeller_input_dict = propeller_model.list_inputs(
            return_format='dict', units=True, out_stream=None, all_procs=True
        )
        propeller_output_dict = propeller_model.list_outputs(
            return_format='dict', units=True, out_stream=None, all_procs=True
        )
        propeller_input_list = list(
            set(
                propeller_input_dict[key]['prom_name']
                for key in propeller_input_dict
                if '.' not in propeller_input_dict[key]['prom_name']
            )
        )
        propeller_output_list = list(
            set(
                propeller_output_dict[key]['prom_name']
                for key in propeller_output_dict
                if '.' not in propeller_output_dict[key]['prom_name']
            )
        )
        propeller_inputs = []
        # always promote all propeller model outputs w/o aliasing except thrust
        propeller_outputs = []

        #########################
        # SHP MODEL CONNECTIONS #
        #########################
        # Everything not explicitly handled here gets promoted later on
        # Thrust outputs are directly connected to thrust adder comp (this is a special case)
        if Dynamic.Vehicle.Propulsion.THRUST in shp_output_list:
            self.connect(
                f'{shp_model.name}.{Dynamic.Vehicle.Propulsion.THRUST}',
                'thrust_summation.turboshaft_thrust',
            )
            shp_output_list.remove(Dynamic.Vehicle.Propulsion.THRUST)

        # Gearbox connections
        if has_gearbox:
            # Cover for several edge cases: connect shp_out to gearbox_in, duplicate shp_out and
            # gearbox_out
            for var in shp_output_list.copy():
                # Check if var is output from both shp_model and gearbox model
                # RPM has special handling, so skip it here
                if var in gearbox_output_list or var + '_out' in gearbox_output_list:
                    shp_output_list.remove(var)
                # if var is shp_output and gearbox input, connect on shp -> gearbox side
                if var + '_in' in gearbox_input_list and var != Dynamic.Vehicle.Propulsion.RPM:
                    gearbox_input_list.remove(var + '_in')
                    self.connect(f'{shp_model.name}.{var}', f'{gearbox_model.name}.{var}_in')
                # otherwise it gets promoted, which will get done later

        ########################
        # RPM SPECIAL HANDLING #
        ########################
        # If fixed RPM is requested by the user, use that value. Override RPM output from shaft
        # power model if present, warning user
        # If no fixed RPM requested, then check if SHP model outputs it - if not, then make promotions/
        # connections from passthrough component, which will get added as a control by the turboprop
        # model if any sub-builders request it as a control
        rpm_source_comp = self._get_subsystem('rpm_source')

        if Aircraft.Engine.FIXED_RPM in aviary_inputs:
            fixed_rpm = aviary_inputs.get_val(Aircraft.Engine.FIXED_RPM, units='rpm')

            if Dynamic.Vehicle.Propulsion.RPM in shp_output_list:
                if aviary_inputs.get_val(Settings.VERBOSITY) >= Verbosity.BRIEF:
                    warnings.warn(
                        f'Overriding RPM value outputted by EngineModel {shp_model.name} with fixed '
                        f'RPM of {fixed_rpm} rpm'
                    )

                shp_outputs.append(
                    (
                        Dynamic.Vehicle.Propulsion.RPM,
                        'INTERNAL_OVERRIDE:' + Dynamic.Vehicle.Propulsion.RPM,
                    )
                )
                shp_output_list.remove(Dynamic.Vehicle.Propulsion.RPM)

            rpm_source_comp.add_output(
                Dynamic.Vehicle.Propulsion.RPM, np.ones(num_nodes) * fixed_rpm, units='rpm'
            )
            # If gearbox exists, don't promote RPM from source because gearbox will modify it
            # If no gearbox, just promote RPM and let OM connect it everywhere
            if has_gearbox and f'{Dynamic.Vehicle.Propulsion.RPM}_in' in gearbox_input_list:
                self.connect(
                    f'rpm_source.{Dynamic.Vehicle.Propulsion.RPM}',
                    f'{gearbox_model.name}.{Dynamic.Vehicle.Propulsion.RPM}_in',
                )
                gearbox_input_list.remove(f'{Dynamic.Vehicle.Propulsion.RPM}_in')
            else:
                self.promotes('rpm_source', ['*'])
            # models such as motor take RPM as input
            if Dynamic.Vehicle.Propulsion.RPM in shp_input_list:
                self.connect(
                    f'rpm_source.{Dynamic.Vehicle.Propulsion.RPM}',
                    f'{shp_model.name}.{Dynamic.Vehicle.Propulsion.RPM}',
                )
                shp_input_list.remove(Dynamic.Vehicle.Propulsion.RPM)
        else:
            if Dynamic.Vehicle.Propulsion.RPM not in shp_output_list:
                # There is no defined source for RPM, so if any component needs it, we use the
                # passthrough component so an "external" source, like a Dymos control, can be connected
                if (
                    Dynamic.Vehicle.Propulsion.RPM in shp_input_list
                    or (
                        has_gearbox and f'{Dynamic.Vehicle.Propulsion.RPM}_in' in gearbox_input_list
                    )
                    or Dynamic.Vehicle.Propulsion.RPM in propeller_input_list
                ):
                    # expose RPM input so Dymos can control it
                    self.promotes(
                        'rpm_source',
                        inputs=[('RPM_control', f'{Dynamic.Vehicle.Propulsion.RPM}_control')],
                    )
            # Go through all sub-systems and connect RPM inputs as needed
            if Dynamic.Vehicle.Propulsion.RPM in shp_input_list:
                self.connect(
                    'rpm_source.RPM',
                    f'{shp_model.name}.{Dynamic.Vehicle.Propulsion.RPM}',
                )
                shp_input_list.remove(Dynamic.Vehicle.Propulsion.RPM)

            if has_gearbox and f'{Dynamic.Vehicle.Propulsion.RPM}_in' in gearbox_input_list:
                if Dynamic.Vehicle.Propulsion.RPM in shp_output_list:
                    self.connect(
                        f'{shp_model.name}.{Dynamic.Vehicle.Propulsion.RPM}',
                        f'{gearbox_model.name}.{Dynamic.Vehicle.Propulsion.RPM}_in',
                    )
                    gearbox_input_list.remove(f'{Dynamic.Vehicle.Propulsion.RPM}_in')
                    shp_output_list.remove(Dynamic.Vehicle.Propulsion.RPM)
                else:
                    self.connect(
                        'rpm_source.RPM',
                        f'{gearbox_model.name}.{Dynamic.Vehicle.Propulsion.RPM}_in',
                    )

                gearbox_input_list.remove(Dynamic.Vehicle.Propulsion.RPM + '_in')

            if Dynamic.Vehicle.Propulsion.RPM in propeller_input_list:
                self.connect(
                    'rpm_source.RPM',
                    f'{propeller_model.name}.{Dynamic.Vehicle.Propulsion.RPM}',
                )
                propeller_input_list.remove(Dynamic.Vehicle.Propulsion.RPM)

        # All other shp model inputs/outputs that don't interact with other components will be promoted
        for var in shp_input_list:
            shp_inputs.append(var)
        for var in shp_output_list:
            shp_outputs.append(var)

        #############################
        # GEARBOX MODEL CONNECTIONS #
        #############################
        if has_gearbox:
            # Promote all inputs which don't come from shp model (those got connected), don't
            # promote ones in skip list
            for var in gearbox_input_list.copy():
                if var not in skipped_inputs and var[-3:] != '_in':
                    gearbox_inputs.append(var)
                # vars in skipped_inputs should never exist in either input list
                gearbox_input_list.remove(var)

            # gearbox outputs always get promoted, also connect to propeller
            for var in propeller_input_list.copy():
                # connect variables with exact name match to propeller
                if var in gearbox_output_list:
                    self.connect(var, f'{propeller_model.name}.{var}')
                    propeller_input_list.remove(var)

            # NOTE a technically better way is to find "_in" and "_out" pairs beforehand and use
            #      that list instead, to avoid catching coincidentally named outputs
            # alias outputs with '_out' to their "base" names, connect matching propeller inputs
            for var in gearbox_output_list.copy():
                if var[-4:] == '_out':
                    new_var = var[:-4]
                    gearbox_outputs.append((var, new_var))
                    gearbox_output_list.remove(var)
                    if new_var in propeller_input_list:
                        self.connect(new_var, f'{propeller_model.name}.{new_var}')
                        propeller_input_list.remove(new_var)

        # inputs/outputs that didn't need special handling get promoted
        for var in gearbox_input_list:
            gearbox_inputs.append(var)
        for var in gearbox_output_list:
            gearbox_outputs.append(var)

        ###############################
        # PROPELLER MODEL CONNECTIONS #
        ###############################
        if Dynamic.Vehicle.Propulsion.THRUST in propeller_output_list:
            self.connect(
                f'{propeller_model.name}.{Dynamic.Vehicle.Propulsion.THRUST}',
                'thrust_summation.propeller_thrust',
            )
            propeller_output_list.remove(Dynamic.Vehicle.Propulsion.THRUST)

        # we will promote all inputs not in skip list
        for var in propeller_input_list:
            if var not in skipped_inputs:
                propeller_inputs.append(var)
        for var in propeller_output_list:
            propeller_outputs.append(var)

        ##############
        # PROMOTIONS #
        ##############
        # bulk promote desired inputs and outputs for each subsystem we have been tracking
        self.promotes(shp_model.name, inputs=shp_inputs, outputs=shp_outputs)

        if has_gearbox:
            self.promotes(gearbox_model.name, inputs=gearbox_inputs, outputs=gearbox_outputs)

        self.promotes(propeller_model_name, inputs=propeller_inputs, outputs=propeller_outputs)
