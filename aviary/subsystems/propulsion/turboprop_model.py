import warnings

import numpy as np
import openmdao.api as om

from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.subsystems.propulsion.engine_model import EngineModel
from aviary.subsystems.propulsion.engine_deck import EngineDeck
from aviary.subsystems.propulsion.utils import EngineModelVariables
from aviary.utils.named_values import NamedValues
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Aircraft, Dynamic, Settings
from aviary.variable_info.enums import Verbosity
from aviary.subsystems.propulsion.propeller.propeller_performance import PropellerPerformance
from aviary.subsystems.propulsion.gearbox.gearbox_builder import GearboxBuilder


class TurbopropModel(EngineModel):
    """
    EngineModel that combines a model for shaft power generation (default is EngineDeck)
    and a model for propeller performance (default is Hamilton Standard).

    Attributes
    ----------
    name : str ('engine')
        Object label.
    options : AviaryValues (<empty>)
        Inputs and options related to engine model.
    data : NamedVaues (<empty>)
        If using an engine deck, engine performance data (optional). If provided, used
        instead of tabular data file.
    shaft_power_model : SubsystemBuilderBase (<empty>)
        Subsystem builder for the shaft power generating component. If None, an
        EngineDeck built using provided options is used.
    propeller_model : SubsystemBuilderBase (<empty>)
        Subsystem builder for the propeller. If None, the Hamilton Standard methodology
        is used to model the propeller.
    gearbox_model : SubsystemBuilderBase (<empty>)
        Subsystem builder used for the gearbox. If None, the simple gearbox model is
        used.

    Methods
    -------
    build_pre_mission
    build_mission
    build_post_mission
    get_val
    set_val
    update
    """

    def __init__(
        self,
        name='turboprop_model',
        options: AviaryValues = None,
        data: NamedValues = None,
        shaft_power_model: SubsystemBuilderBase = None,
        propeller_model: SubsystemBuilderBase = None,
        gearbox_model: SubsystemBuilderBase = None,
    ):

        # also calls _preprocess_inputs() as part of EngineModel __init__
        super().__init__(name, options)

        self.shaft_power_model = shaft_power_model
        self.propeller_model = propeller_model
        self.gearbox_model = gearbox_model

        # Initialize turboshaft engine deck. New required variable set w/o thrust
        if shaft_power_model is None:
            self.shaft_power_model = EngineDeck(
                name=name + '_engine_deck',
                options=options,
                data=data,
                required_variables={
                    EngineModelVariables.ALTITUDE,
                    EngineModelVariables.MACH,
                    EngineModelVariables.THROTTLE,
                },
            )

        # TODO No reason gearbox model needs to be required. All connections can
        # be handled in configure - need to figure out when user wants gearbox without
        # directly passing builder
        if gearbox_model is None:
            # TODO where can we bring in include_constraints? kwargs in init is an option,
            # but that still requires the L2 interface
            self.gearbox_model = GearboxBuilder(
                name=name + '_gearbox', include_constraints=True
            )

    # BUG if using both custom subsystems that happen to share a kwarg but need different values, this breaks
    def build_pre_mission(self, aviary_inputs, **kwargs) -> om.Group:
        shp_model = self.shaft_power_model
        propeller_model = self.propeller_model
        gearbox_model = self.gearbox_model
        turboprop_group = om.Group()

        # TODO engine scaling for turboshafts requires EngineSizing to be refactored to
        # accept target scaling variable as an option, skipping for now
        if type(shp_model) is not EngineDeck:
            shp_model_pre_mission = shp_model.build_pre_mission(aviary_inputs, **kwargs)
            if shp_model_pre_mission is not None:
                turboprop_group.add_subsystem(
                    shp_model_pre_mission.name,
                    subsys=shp_model_pre_mission,
                    promotes=['*']
                )

        gearbox_model_pre_mission = gearbox_model.build_pre_mission(
            aviary_inputs, **kwargs
        )
        if gearbox_model_pre_mission is not None:
            turboprop_group.add_subsystem(
                gearbox_model_pre_mission.name,
                subsys=gearbox_model_pre_mission,
                promotes=['*'],
            )

        if propeller_model is not None:
            propeller_model_pre_mission = propeller_model.build_pre_mission(
                aviary_inputs, **kwargs
            )
            if propeller_model_pre_mission is not None:
                turboprop_group.add_subsystem(
                    propeller_model_pre_mission.name,
                    subsys=propeller_model_pre_mission,
                    promotes=['*']
                )

        return turboprop_group

    def build_mission(self, num_nodes, aviary_inputs, **kwargs):
        turboprop_group = TurbopropMission(
            num_nodes=num_nodes,
            shaft_power_model=self.shaft_power_model,
            propeller_model=self.propeller_model,
            gearbox_model=self.gearbox_model,
            aviary_inputs=aviary_inputs,
            kwargs=kwargs,
        )

        return turboprop_group

    def build_post_mission(self, aviary_inputs, **kwargs):
        shp_model = self.shaft_power_model
        gearbox_model = self.gearbox_model
        propeller_model = self.propeller_model
        turboprop_group = om.Group()

        shp_model_post_mission = shp_model.build_post_mission(aviary_inputs, **kwargs)
        if shp_model_post_mission is not None:
            turboprop_group.add_subsystem(
                shp_model.name,
                subsys=shp_model_post_mission,
                aviary_options=aviary_inputs,
            )

        gearbox_model_post_mission = gearbox_model.build_post_mission(
            aviary_inputs, **kwargs
        )
        if gearbox_model_post_mission is not None:
            turboprop_group.add_subsystem(
                gearbox_model.name,
                subsys=gearbox_model_post_mission,
                aviary_options=aviary_inputs,
            )

        if propeller_model is not None:
            propeller_model_post_mission = propeller_model.build_post_mission(
                aviary_inputs, **kwargs
            )
            if propeller_model_post_mission is not None:
                turboprop_group.add_subsystem(
                    propeller_model.name,
                    subsys=propeller_model_post_mission,
                    aviary_options=aviary_inputs,
                )

        return turboprop_group


class TurbopropMission(om.Group):
    def initialize(self):
        self.options.declare(
            'num_nodes', types=int, desc='Number of nodes to be evaluated in the RHS'
        )
        self.options.declare('shaft_power_model', desc='shaft power generation model')
        self.options.declare('propeller_model', desc='propeller model')
        self.options.declare('gearbox_model', desc='gearbox model')
        self.options.declare('kwargs', desc='kwargs for turboprop mission model')
        self.options.declare(
            'aviary_inputs', desc='aviary inputs for turboprop mission model'
        )

    def setup(self):
        # All promotions for configurable components in this group are handled during
        # configure()

        # save num_nodes for use in configure()
        self.num_nodes = num_nodes = self.options['num_nodes']
        shp_model = self.options['shaft_power_model']
        propeller_model = self.options['propeller_model']
        gearbox_model = self.options['gearbox_model']
        kwargs = self.options['kwargs']
        # save aviary_inputs for use in configure()
        self.aviary_inputs = aviary_inputs = self.options['aviary_inputs']

        # Shaft Power Model
        try:
            shp_kwargs = kwargs[shp_model.name]
        except (AttributeError, KeyError):
            shp_kwargs = {}
        shp_model_mission = shp_model.build_mission(
            num_nodes, aviary_inputs, **shp_kwargs)
        if shp_model_mission is not None:
            self.add_subsystem(shp_model.name, subsys=shp_model_mission)

        # NOTE: this subsystem is a empty component that has fixed RPM added as an output
        #       in configure() if provided in aviary_inputs
        self.add_subsystem('fixed_rpm_source', subsys=om.IndepVarComp())

        # Gearbox Model
        try:
            gearbox_kwargs = kwargs[gearbox_model.name]
        except (AttributeError, KeyError):
            gearbox_kwargs = {}
        if gearbox_model is not None:
            gearbox_model_mission = gearbox_model.build_mission(
                num_nodes, aviary_inputs, **gearbox_kwargs
            )
            if gearbox_model_mission is not None:
                self.add_subsystem(gearbox_model.name, subsys=gearbox_model_mission)

        # Propeller Model
        try:
            propeller_kwargs = kwargs[propeller_model.name]
        except (AttributeError, KeyError):
            propeller_kwargs = {}
        if propeller_model is not None:
            propeller_group = om.Group()
            propeller_model_mission = propeller_model.build_mission(
                num_nodes, aviary_inputs, **propeller_kwargs
            )
            if propeller_model_mission is not None:
                propeller_group.add_subsystem(
                    propeller_model.name + '_base',
                    subsys=propeller_model_mission,
                    promotes_inputs=['*'],
                    promotes_outputs=[Dynamic.Mission.THRUST],
                )

                propeller_model_mission_max = propeller_model.build_mission(
                    num_nodes, aviary_inputs, **propeller_kwargs
                )
                propeller_group.add_subsystem(
                    propeller_model.name + '_max',
                    subsys=propeller_model_mission_max,
                    promotes_inputs=[
                        '*',
                        (Dynamic.Mission.SHAFT_POWER, Dynamic.Mission.SHAFT_POWER_MAX),
                    ],
                    promotes_outputs=[
                        (Dynamic.Mission.THRUST, Dynamic.Mission.THRUST_MAX)
                    ],
                )

            self.add_subsystem(propeller_model.name, propeller_group)

        else:
            # use the Hamilton Standard model
            # only promote top-level inputs to avoid conflicts with max group
            prop_inputs = [
                Dynamic.Mission.MACH,
                Aircraft.Engine.PROPELLER_TIP_SPEED_MAX,
                Aircraft.Engine.PROPELLER_TIP_MACH_MAX,
                Dynamic.Mission.DENSITY,
                Dynamic.Mission.VELOCITY,
                Aircraft.Engine.PROPELLER_DIAMETER,
                Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR,
                Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICIENT,
                Aircraft.Nacelle.AVG_DIAMETER,
                Dynamic.Mission.SPEED_OF_SOUND,
                Dynamic.Mission.RPM,
            ]
            try:
                propeller_kwargs = kwargs['hamilton_standard']
            except KeyError:
                propeller_kwargs = {}

            propeller_group = om.Group()

            propeller_group.add_subsystem(
                'propeller_model_base',
                PropellerPerformance(
                    aviary_options=aviary_inputs,
                    num_nodes=num_nodes,
                    **propeller_kwargs,
                ),
                promotes=['*'],
            )

            propeller_group.add_subsystem(
                'propeller_model_max',
                PropellerPerformance(
                    aviary_options=aviary_inputs,
                    num_nodes=num_nodes,
                    **propeller_kwargs,
                ),
                promotes_inputs=[
                    *prop_inputs,
                    (Dynamic.Mission.SHAFT_POWER, Dynamic.Mission.SHAFT_POWER_MAX),
                ],
                promotes_outputs=[(Dynamic.Mission.THRUST, Dynamic.Mission.THRUST_MAX)],
            )

            self.add_subsystem('propeller_model', propeller_group)

        thrust_adder = om.ExecComp(
            'turboprop_thrust=turboshaft_thrust+propeller_thrust',
            turboprop_thrust={'val': np.zeros(num_nodes), 'units': 'lbf'},
            turboshaft_thrust={'val': np.zeros(num_nodes), 'units': 'lbf'},
            propeller_thrust={'val': np.zeros(num_nodes), 'units': 'lbf'},
            has_diag_partials=True,
        )

        max_thrust_adder = om.ExecComp(
            'turboprop_thrust_max=turboshaft_thrust_max+propeller_thrust_max',
            turboprop_thrust_max={'val': np.zeros(num_nodes), 'units': 'lbf'},
            turboshaft_thrust_max={'val': np.zeros(num_nodes), 'units': 'lbf'},
            propeller_thrust_max={'val': np.zeros(num_nodes), 'units': 'lbf'},
            has_diag_partials=True,
        )

        self.add_subsystem(
            'thrust_adder',
            subsys=thrust_adder,
            promotes_inputs=['*'],
            promotes_outputs=[('turboprop_thrust', Dynamic.Mission.THRUST)],
        )

        self.add_subsystem(
            'max_thrust_adder',
            subsys=max_thrust_adder,
            promotes_inputs=['*'],
            promotes_outputs=[('turboprop_thrust_max', Dynamic.Mission.THRUST_MAX)],
        )

    def configure(self):
        """
        Correctly connect variables between shaft power model, gearbox, and propeller,
        aliasing names if they are present in both sets of connections.

        If a gearbox is present, inputs to the gearbox are usually done via connection,
        while outputs from the gearbox are promoted. This prevents intermediate values
        from "leaking" out of the model and getting incorrectly connected to outside
        components. It is assumed only the gearbox has variables like this.

        Set up fixed RPM value if requested by user, which overrides any RPM defined by
        shaft power model
        """
        has_gearbox = self.options['gearbox_model'] is not None

        # TODO this list shouldn't be hardcoded - it should mirror propulsion_mission list
        # Don't promote inputs that are in this list - shaft power should be an output
        #   of this system, also having it as an input causes feedback loop problem at
        #   the propulsion level
        skipped_inputs = [
            Dynamic.Mission.ELECTRIC_POWER_IN,
            Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE,
            Dynamic.Mission.NOX_RATE,
            Dynamic.Mission.SHAFT_POWER,
            Dynamic.Mission.SHAFT_POWER_MAX,
            Dynamic.Mission.TEMPERATURE_T4,
            Dynamic.Mission.THRUST,
            Dynamic.Mission.THRUST_MAX,
        ]

        # Build lists of inputs/outputs for each component as needed:
        # "_input_list" or "_output_list" are all variables that still need to be
        #   connected or promoted. This list is pared down as each variable is handled.
        # "_inputs" or "_outputs" is a list that tracks all the pomotions needed for a
        #   given component, which is done at the end as a bulk promote.

        shp_model = self._get_subsystem(self.options['shaft_power_model'].name)
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
        # always promote all shaft power model inputs w/o aliasing
        shp_inputs = ['*']
        shp_outputs = []

        if has_gearbox:
            gearbox_model = self._get_subsystem(self.options['gearbox_model'].name)
            gearbox_input_dict = gearbox_model.list_inputs(
                return_format='dict', units=True, out_stream=None, all_procs=True
            )
            # Assumption is made that variables with '_out' should never be promoted or
            #   connected as top-level input to gearbox. This is necessary because
            #   Aviary gearbox uses things like shp_out internally, like when computing
            #   torque output, so "shp_out" is an input to that internal component
            gearbox_input_list = list(
                set(
                    gearbox_input_dict[key]['prom_name']
                    for key in gearbox_input_dict
                    if '.' not in gearbox_input_dict[key]['prom_name']
                    and '_out' not in gearbox_input_dict[key]['prom_name']
                )
            )
            gearbox_inputs = []
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
            gearbox_outputs = []

        if self.options['propeller_model'] is None:
            propeller_model_name = 'propeller_model'
        else:
            propeller_model_name = self.options['propeller_model'].name
        propeller_model = self._get_subsystem(propeller_model_name)
        propeller_input_dict = propeller_model.list_inputs(
            return_format='dict', units=True, out_stream=None, all_procs=True
        )
        propeller_input_list = list(
            set(
                propeller_input_dict[key]['prom_name']
                for key in propeller_input_dict
                if '.' not in propeller_input_dict[key]['prom_name']
            )
        )
        propeller_inputs = []
        # always promote all propeller model outputs w/o aliasing except thrust
        propeller_outputs = [
            '*',
            (Dynamic.Mission.THRUST, 'propeller_thrust'),
            (Dynamic.Mission.THRUST_MAX, 'propeller_thrust_max'),
        ]

        #########################
        # SHP MODEL CONNECTIONS #
        #########################
        # Everything not explicitly handled here gets promoted later on
        # Thrust outputs are directly promoted with alias (this is a special case)
        if Dynamic.Mission.THRUST in shp_output_list:
            shp_outputs.append((Dynamic.Mission.THRUST, 'turboshaft_thrust'))
            shp_output_list.remove(Dynamic.Mission.THRUST)

        if Dynamic.Mission.THRUST_MAX in shp_output_list:
            shp_outputs.append((Dynamic.Mission.THRUST_MAX, 'turboshaft_thrust_max'))
            shp_output_list.remove(Dynamic.Mission.THRUST_MAX)

        # Gearbox connections
        if has_gearbox:
            for var in shp_output_list.copy():
                # Check for case: var is output from shp_model, connects to gearbox, then
                #   gets updated by gearbox
                # RPM has special handling, so skip it here
                if var + '_in' in gearbox_input_list and var != Dynamic.Mission.RPM:
                    # if var is in gearbox input and output, connect on shp -> gearbox side
                    if (
                        var in gearbox_output_list
                        or var + '_out' in gearbox_output_list
                    ):
                        shp_outputs.append((var, var + '_gearbox'))
                        shp_output_list.remove(var)
                        gearbox_inputs.append((var + '_in', var + '_gearbox'))
                        gearbox_input_list.remove(var + '_in')
                    # otherwise it gets promoted, which will get done later

        # If fixed RPM is requested by the user, use that value. Override RPM output
        #   from shaft power model if present, warning user
        rpm_ivc = self._get_subsystem('fixed_rpm_source')

        if Aircraft.Engine.FIXED_RPM in self.aviary_inputs:
            fixed_rpm = self.aviary_inputs.get_val(
                Aircraft.Engine.FIXED_RPM, units='rpm'
            )

            if Dynamic.Mission.RPM in shp_output_list:
                if self.aviary_inputs.get_val(Settings.VERBOSITY) >= Verbosity.BRIEF:
                    warnings.warn(
                        'Overriding RPM value outputted by EngineModel'
                        f'{shp_model.name} with fixed RPM of {fixed_rpm}'
                    )

                shp_outputs.append(
                    (Dynamic.Mission.RPM, 'AUTO_OVERRIDE:' + Dynamic.Mission.RPM)
                )
                shp_output_list.remove(Dynamic.Mission.RPM)

            fixed_rpm_nn = np.ones(self.num_nodes) * fixed_rpm

            rpm_ivc.add_output(Dynamic.Mission.RPM, fixed_rpm_nn, units='rpm')
            if has_gearbox:
                self.promotes('fixed_rpm_source', [(Dynamic.Mission.RPM, 'fixed_rpm')])
                gearbox_inputs.append((Dynamic.Mission.RPM + '_in', 'fixed_rpm'))
                gearbox_input_list.remove(Dynamic.Mission.RPM + '_in')
            else:
                self.promotes('fixed_rpm_source', ['*'])
        else:
            rpm_ivc.add_output('AUTO_OVERRIDE:' + Dynamic.Mission.RPM, 1.0, units='rpm')
            if has_gearbox:
                if Dynamic.Mission.RPM in shp_output_list:
                    shp_outputs.append(
                        (Dynamic.Mission.RPM, Dynamic.Mission.RPM + '_gearbox')
                    )
                    shp_output_list.remove(Dynamic.Mission.RPM)
                gearbox_inputs.append(
                    (Dynamic.Mission.RPM + '_in', Dynamic.Mission.RPM + '_gearbox')
                )
                gearbox_input_list.remove(Dynamic.Mission.RPM + '_in')

        # All other shp model outputs that don't interact with gearbox will be promoted
        for var in shp_output_list:
            shp_outputs.append(var)

        #############################
        # GEARBOX MODEL CONNECTIONS #
        #############################
        if has_gearbox:
            # Promote all inputs which don't come from shp model (those got connected),
            #   don't promote ones in skip list
            for var in gearbox_input_list.copy():
                if var not in skipped_inputs:
                    gearbox_inputs.append(var)
                # DO NOT promote inputs in skip list - always skip
                gearbox_input_list.remove(var)

            # gearbox outputs can always get promoted
            for var in propeller_input_list.copy():
                if var in gearbox_output_list and var in propeller_input_list:
                    gearbox_outputs.append((var, var))
                    gearbox_output_list.remove(var)
                    # connect variables in skip list to propeller
                    if var in skipped_inputs:
                        self.connect(
                            var,
                            propeller_model.name + '.' + var,
                        )

                # alias outputs with 'out' to match with propeller
                if var + '_out' in gearbox_output_list and var in propeller_input_list:
                    gearbox_outputs.append((var + '_out', var))
                    gearbox_output_list.remove(var + '_out')
                    # connect variables in skip list to propeller
                    if var in skipped_inputs:
                        self.connect(
                            var,
                            propeller_model.name + '.' + var,
                        )

        # inputs/outputs that didn't need special handling will get promoted
        for var in gearbox_input_list:
            gearbox_inputs.append(var)
        for var in gearbox_output_list:
            gearbox_outputs.append(var)

        ###############################
        # PROPELLER MODEL CONNECTIONS #
        ###############################
        # we will promote all inputs not in skip list
        for var in propeller_input_list.copy():
            if var not in skipped_inputs:
                propeller_inputs.append(var)
            propeller_input_list.remove(var)

        ##############
        # PROMOTIONS #
        ##############
        # bulk promote desired inputs and outputs for each subsystem we have been tracking
        self.promotes(shp_model.name, inputs=shp_inputs, outputs=shp_outputs)

        if has_gearbox:
            self.promotes(
                gearbox_model.name, inputs=gearbox_inputs, outputs=gearbox_outputs
            )

        self.promotes(
            propeller_model_name, inputs=propeller_inputs, outputs=propeller_outputs
        )
