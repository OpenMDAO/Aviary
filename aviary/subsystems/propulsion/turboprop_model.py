import numpy as np
import openmdao.api as om

from aviary.subsystems.propulsion.engine_model import EngineModel
from aviary.subsystems.propulsion.engine_deck import EngineDeck
from aviary.subsystems.propulsion.utils import EngineModelVariables
from aviary.utils.named_values import NamedValues
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Aircraft, Dynamic
from aviary.subsystems.propulsion.propeller.propeller_performance import PropellerPerformance


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

    Methods
    -------
    build_pre_mission
    build_mission
    build_post_mission
    get_val
    set_val
    update
    """

    def __init__(self, name='turboprop_model', options: AviaryValues = None,
                 data: NamedValues = None, shaft_power_model=None, propeller_model=None):

        # also calls _preprocess_inputs() as part of EngineModel __init__
        super().__init__(name, options)

        self.shaft_power_model = shaft_power_model
        self.propeller_model = propeller_model

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

    # BUG if using both custom subsystems that happen to share a kwarg but need different values, this breaks
    def build_pre_mission(self, aviary_inputs, **kwargs) -> om.Group:
        shp_model = self.shaft_power_model
        propeller_model = self.propeller_model
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
            aviary_inputs=aviary_inputs,
            kwargs=kwargs,
        )

        return turboprop_group

    def build_post_mission(self, aviary_inputs, **kwargs):
        shp_model = self.shaft_power_model
        propeller_model = self.propeller_model
        turboprop_group = om.Group()
        if type(shp_model) is not EngineDeck:
            shp_model_post_mission = shp_model.build_post_mission(
                aviary_inputs, **kwargs
            )
            if shp_model_post_mission is not None:
                turboprop_group.add_subsystem(
                    shp_model_post_mission.name,
                    subsys=shp_model_post_mission,
                    aviary_options=aviary_inputs,
                )

        if self.propeller_model is not None:
            propeller_model_post_mission = propeller_model.build_post_mission(
                aviary_inputs, **kwargs
            )
            if propeller_model_post_mission is not None:
                turboprop_group.add_subsystem(
                    propeller_model_post_mission.name,
                    subsys=propeller_model_post_mission,
                    aviary_options=aviary_inputs,
                )

        # turboprop_group.set_input_default(
        #     Aircraft.Engine.PROPELLER_TIP_SPEED_MAX, val=0.0, units='ft/s'
        # )

        return turboprop_group


class TurbopropMission(om.Group):
    def initialize(self):
        self.options.declare(
            'num_nodes', types=int, desc='Number of nodes to be evaluated in the RHS'
        )
        self.options.declare('shaft_power_model', desc='shaft power generation model')
        self.options.declare('propeller_model', desc='propeller model')
        self.options.declare('kwargs', desc='kwargs for turboprop mission models')
        self.options.declare(
            'aviary_inputs', desc='aviary inputs for turboprop mission'
        )

    def setup(self):
        num_nodes = self.options['num_nodes']
        shp_model = self.options['shaft_power_model']
        propeller_model = self.options['propeller_model']
        kwargs = self.options['kwargs']
        aviary_inputs = self.options['aviary_inputs']

        max_thrust_group = om.Group()

        try:
            shp_kwargs = kwargs[shp_model.name]
        except (AttributeError, KeyError):
            shp_kwargs = {}
        shp_model_mission = shp_model.build_mission(
            num_nodes, aviary_inputs, **shp_kwargs)
        if shp_model_mission is not None:
            self.add_subsystem(
                shp_model.name,
                subsys=shp_model_mission,
                promotes_inputs=['*'],
            )

        # Gearbox can go here

        try:
            propeller_kwargs = kwargs[propeller_model.name]
        except (AttributeError, KeyError):
            propeller_kwargs = {}
        if propeller_model is not None:

            propeller_model_mission = propeller_model.build_mission(
                num_nodes, aviary_inputs, **propeller_kwargs
            )
            if propeller_model_mission is not None:
                self.add_subsystem(
                    propeller_model.name,
                    subsys=propeller_model_mission,
                    promotes_inputs=[
                        '*',
                        (Dynamic.Mission.SHAFT_POWER, 'propeller_shaft_power'),
                    ],
                    promotes_outputs=[
                        '*',
                        (Dynamic.Mission.THRUST, 'propeller_thrust'),
                    ],
                )

                self.connect(Dynamic.Mission.SHAFT_POWER, 'propeller_shaft_power')

                propeller_model_mission_max = propeller_model.build_mission(
                    num_nodes, aviary_inputs, **propeller_kwargs
                )
                max_thrust_group.add_subsystem(
                    propeller_model.name + '_max',
                    subsys=propeller_model_mission_max,
                    promotes_inputs=['*',
                                     (Dynamic.Mission.SHAFT_POWER, 'propeller_shaft_power_max')],
                    promotes_outputs=[(Dynamic.Mission.THRUST, 'propeller_thrust_max')]
                )

                self.connect(
                    Dynamic.Mission.SHAFT_POWER_MAX, 'propeller_shaft_power_max'
                )

        else:  # use the Hamilton Standard model
            # only promote top-level inputs to avoid conflicts with max group
            prop_inputs = [
                Dynamic.Mission.MACH,
                Aircraft.Engine.PROPELLER_TIP_SPEED_MAX,
                Dynamic.Mission.DENSITY,
                Dynamic.Mission.VELOCITY,
                Aircraft.Engine.PROPELLER_DIAMETER,
                Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR,
                Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICIENT,
                Aircraft.Nacelle.AVG_DIAMETER,
                Dynamic.Mission.SPEED_OF_SOUND,
            ]
            try:
                propeller_kwargs = kwargs['hamilton_standard']
            except KeyError:
                propeller_kwargs = {}

            self.add_subsystem(
                'propeller_model',
                PropellerPerformance(
                    aviary_options=aviary_inputs,
                    num_nodes=num_nodes,
                    **propeller_kwargs,
                ),
                promotes_inputs=[
                    *prop_inputs,
                    (Dynamic.Mission.SHAFT_POWER, 'propeller_shaft_power'),
                ],
                promotes_outputs=[
                    '*',
                    (Dynamic.Mission.THRUST, 'propeller_thrust'),
                ],
            )

            self.connect(Dynamic.Mission.SHAFT_POWER, 'propeller_shaft_power')

            max_thrust_group.add_subsystem(
                'propeller_model_max',
                PropellerPerformance(
                    aviary_options=aviary_inputs,
                    num_nodes=num_nodes,
                    **propeller_kwargs,
                ),
                promotes_inputs=[
                    *prop_inputs,
                    (Dynamic.Mission.SHAFT_POWER, 'propeller_shaft_power_max'),
                ],
                promotes_outputs=[(Dynamic.Mission.THRUST, 'propeller_thrust_max')],
            )

            self.connect(Dynamic.Mission.SHAFT_POWER_MAX, 'propeller_shaft_power_max')

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

        max_thrust_group.add_subsystem(
            'max_thrust_adder',
            subsys=max_thrust_adder,
            promotes_inputs=['*'],
            promotes_outputs=[('turboprop_thrust_max', Dynamic.Mission.THRUST_MAX)]
        )

        self.add_subsystem(
            'turboprop_max_group',
            max_thrust_group,
            promotes_inputs=['*'],
            promotes_outputs=[Dynamic.Mission.THRUST_MAX],
        )

    def configure(self):
        # configure step to alias thrust output from shaft power model if present
        shp_model = self._get_subsystem(self.options['shaft_power_model'].name)
        output_dict = shp_model.list_outputs(
            return_format='dict', units=True, out_stream=None, all_procs=True
        )

        outputs = ['*']

        if Dynamic.Mission.THRUST in [
            output_dict[key]['prom_name'] for key in output_dict
        ]:
            outputs.append((Dynamic.Mission.THRUST, 'turboshaft_thrust'))

        if Dynamic.Mission.THRUST_MAX in [
            output_dict[key]['prom_name'] for key in output_dict
        ]:
            outputs.append((Dynamic.Mission.THRUST_MAX, 'turboshaft_thrust_max'))

        self.promotes(shp_model.name, outputs=outputs)
