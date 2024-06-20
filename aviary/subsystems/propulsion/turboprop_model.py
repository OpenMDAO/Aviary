import numpy as np
import openmdao.api as om

from aviary.subsystems.propulsion.engine_model import EngineModel
from aviary.subsystems.propulsion.engine_deck import EngineDeck
from aviary.subsystems.propulsion.utils import EngineModelVariables
from aviary.utils.named_values import NamedValues
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Aircraft, Dynamic
from aviary.subsystems.propulsion.propeller_performance import PropellerPerformance
from aviary.subsystems.propulsion.utils import UncorrectData
from aviary.mission.gasp_based.flight_conditions import FlightConditions
from aviary.variable_info.enums import SpeedType


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
    shaft_power_model : dict (<empty>)
        If not using an EngineDeck, use this dictionary of OpenMDAO systems.
        Accepted keys are "pre-mission", "mission", and "post-mission". The values in the
        dict are the systems that will be added during the matching method call.
    propeller_model : dict (<empty>)
        If not using the Hamilton Standard model, use this dictionary of OpenMDAO systems.
        Accepted keys are "pre-mission", "mission", and "post-mission". The values in the
        dict are the systems that will be added during the matching method call.

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
            self.shaft_power_model = EngineDeck(name=name + '_engine_deck',
                                                options=options,
                                                data=data,
                                                required_variables={EngineModelVariables.ALTITUDE,
                                                                    EngineModelVariables.MACH,
                                                                    EngineModelVariables.THROTTLE})
            # Manually check if shaft horsepower was provided (can't flag as required variable since either version
            # is acceptable)
            if not self.shaft_power_model.use_shaft_power:
                # repurpose custom error message code from EngineDeck
                # custom error messages depending on data type
                if self.shaft_power_model.read_from_file:
                    message = f'<{self.shaft_power_model.get_val(Aircraft.Engine.DATA_FILE)}>'
                else:
                    message = f'EngineDeck for <{self.name}>'
                raise UserWarning(
                    f'No shaft horsepower variable was provided in {message}')

    # BUG if using both custom subsystems that happen to share a kwarg but need different values, this breaks
    def build_pre_mission(self, aviary_inputs, **kwargs) -> om.Group:
        shp_model = self.shaft_power_model
        prop_model = self.propeller_model
        turboprop_group = om.Group()
        # TODO engine scaling for turboshafts requires EngineSizing to be refactored to
        # accept target scaling variable as an option, skipping for now
        if type(shp_model) is not EngineDeck:
            shp_model_pre_mission = shp_model.build_pre_mission(aviary_inputs, **kwargs)
            if shp_model_pre_mission is not None:
                turboprop_group.add_subsystem(shp_model_pre_mission.name,
                                              subsys=shp_model_pre_mission,
                                              promotes=['*'])

        if prop_model is not None:
            prop_model_pre_mission = prop_model.build_pre_mission(
                aviary_inputs, **kwargs)
            if prop_model_pre_mission is not None:
                turboprop_group.add_subsystem(prop_model_pre_mission.name,
                                              subsys=prop_model_pre_mission,
                                              promotes=['*'])

        return turboprop_group

    def build_mission(self, num_nodes, aviary_inputs, **kwargs):
        shp_model = self.shaft_power_model
        prop_model = self.propeller_model
        turboprop_group = om.Group()

        shp_model_mission = shp_model.build_mission(num_nodes, aviary_inputs, **kwargs)
        if shp_model_mission is not None:
            turboprop_group.add_subsystem(shp_model.name,
                                          subsys=shp_model_mission,
                                          promotes_inputs=['*'],
                                          promotes_outputs=['*', (Dynamic.Mission.THRUST, 'turboshaft_thrust')])

        # ensure uncorrected shaft horsepower is avaliable
        # TODO also make sure corrected is avaliable
        # TODO see if this can be done for non-EngineDecks
        if type(shp_model_mission) is EngineDeck:
            if EngineModelVariables.SHAFT_POWER not in shp_model_mission.engine_variables:
                turboprop_group.add_subsystem('uncorrect_shaft_power',
                                              subsys=UncorrectData(num_nodes=num_nodes,
                                                                   aviary_options=self.options),
                                              promotes_inputs=[('corrected_data', Dynamic.Mission.SHAFT_POWER_CORRECTED),
                                                               Dynamic.Mission.TEMPERATURE,
                                                               Dynamic.Mission.STATIC_PRESSURE,
                                                               Dynamic.Mission.MACH],
                                              promotes_outputs=[('uncorrected_data', Dynamic.Mission.SHAFT_POWER)]),

        if prop_model is not None:  # must assume user-provide propeller group has everything it needs
            prop_model_mission = prop_model.build_mission(
                num_nodes, self.options, **kwargs)
            if prop_model_mission is not None:
                turboprop_group.add_subsystem(prop_model.name,
                                              subsys=prop_model_mission,
                                              promotes_inputs=['*'],
                                              promotes_outputs=['*'])

        else:  # use the Hamilton Standard model
            # calculate atmospheric properties
            # TODO these properties should always just be avaliable across Aviary
            turboprop_group.add_subsystem('flight_conditions',
                                          FlightConditions(num_nodes=num_nodes,
                                                           input_speed_type=SpeedType.MACH),
                                          promotes_inputs=['rho',
                                                           Dynamic.Mission.SPEED_OF_SOUND,
                                                           Dynamic.Mission.MACH],
                                          promotes_outputs=[Dynamic.Mission.DYNAMIC_PRESSURE,
                                                            'EAS',
                                                            ('TAS', 'velocity')])
            # Hamilton Standard method
            turboprop_group.add_subsystem('propeller_model',
                                          PropellerPerformance(aviary_options=self.options,
                                                               num_nodes=num_nodes),
                                          promotes_inputs=['*'],
                                          promotes_outputs=['*'])

        thrust_adder = om.ExecComp('turboprop_thrust=turboshaft_thrust+propeller_thrust',
                                   turboprop_thrust={'val': np.zeros(
                                       num_nodes), 'units': 'lbf'},
                                   turboshaft_thrust={'val': np.zeros(
                                       num_nodes), 'units': 'lbf'},
                                   propeller_thrust={'val': np.zeros(
                                       num_nodes), 'units': 'lbf'},
                                   has_diag_partials=True
                                   )

        turboprop_group.add_subsystem('thrust_adder',
                                      subsys=thrust_adder,
                                      promotes_inputs=['*'],
                                      promotes_outputs=[('turboprop_thrust',
                                                         Dynamic.Mission.THRUST)])

        return turboprop_group

    def build_post_mission(self, aviary_inputs, **kwargs):
        shp_model = self.shaft_power_model
        prop_model = self.propeller_model
        turboprop_group = om.Group()
        if type(shp_model) is not EngineDeck:
            shp_model_post_mission = shp_model.build_post_mission(
                aviary_inputs, **kwargs)
            if shp_model_post_mission is not None:
                turboprop_group.add_subsystem(shp_model_post_mission.name,
                                              subsys=shp_model_post_mission,
                                              aviary_options=aviary_inputs,)

        if self.propeller_model is not None:
            prop_model_post_mission = prop_model.build_mission(aviary_inputs, **kwargs)
            if prop_model_post_mission is not None:
                turboprop_group.add_subsystem(prop_model_post_mission.name,
                                              subsys=prop_model_post_mission,
                                              aviary_options=aviary_inputs,)

        return turboprop_group
