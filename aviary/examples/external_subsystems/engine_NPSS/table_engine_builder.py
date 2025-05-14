import os

import numpy as np
import openmdao.api as om

from aviary.examples.external_subsystems.engine_NPSS.engine_variable_meta_data import (
    ExtendedMetaData,
)
from aviary.examples.external_subsystems.engine_NPSS.engine_variables import Aircraft, Dynamic
from aviary.examples.external_subsystems.engine_NPSS.NPSS_Model.DesignEngineGroup import (
    DesignEngineGroup,
)
from aviary.examples.external_subsystems.engine_NPSS.table_engine_connected_variables import (
    vars_to_connect,
)
from aviary.subsystems.propulsion.engine_model import EngineModel
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.functions import get_aviary_resource_path


class TableEngineBuilder(EngineModel):
    """
    NPSS engine builder from table.

    Attributes
    ----------
    name : str ('NPSS_prop_system')
        object label
    aviary_inputs: AviaryValues
        default Aviary options

    Methods
    -------
    __init__(self, name='NPSS_prop_system', aviary_inputs=AviaryValues()):
        Initializes the TableEngineBuilder object with a given name.
    build_pre_mission(self, aviary_inputs=AviaryValues()) -> openmdao.core.System:
        Builds an OpenMDAO system for the pre-mission computations of the subsystem.
    build_mission(self, num_nodes, aviary_inputs) -> openmdao.core.System:
        Builds an OpenMDAO system for the mission computations of the subsystem.
    get_pre_mission_bus_variables(self) -> dict:
        Transfer training data from pre-mission to mission.
    get_controls(self, phase_name) -> dict:
        Builds dictionary of controls for engine off-design.
    get_design_vars(self) -> dict:
        Builds dictionary of design variables for Engine off-design.
    """

    def __init__(self, name='NPSS_prop_system', aviary_inputs=AviaryValues()):
        super().__init__(name, options=aviary_inputs, meta_data=ExtendedMetaData)

    def build_pre_mission(self, aviary_inputs=AviaryValues()):
        """
        Builds the design (pre-mission) engine model.

        Parameters
        ----------
        aviary_inputs : AviaryValues
            Inputs to Aviary model.

        Returns
        -------
        prob : openmdao.core.Group
            engine model for design.
        """
        return DesignEngineGroup()

    def build_mission(self, num_nodes, aviary_inputs):
        """
        Builds the off-design (mission) engine model.

        Parameters
        ----------
        num_nodes : integer
            Number of points to be evaluated.
        aviary_inputs : AviaryValues
            Inputs to Aviary model.

        Returns
        -------
        prob : openmdao.core.Group
            engine model for off-design.
        """
        interp_method = aviary_inputs.get_val(Aircraft.Engine.INTERPOLATION_METHOD)[0]

        # interpolator object for engine data
        engine = om.MetaModelSemiStructuredComp(
            method=interp_method,
            extrapolate=True,
            vec_size=num_nodes,
            training_data_gradients=True,
        )

        ref = os.path.join(
            'examples',
            'external_subsystems',
            'engine_NPSS',
            'NPSS_Model',
            'Output',
            'RefEngine.outputAviary',
        )
        csv_path = get_aviary_resource_path(ref)
        engine_data = np.genfromtxt(csv_path, skip_header=0)

        # Sort the data by Mach, then altitude, then throttle
        engine_data = engine_data[
            np.lexsort((engine_data[:, 2], engine_data[:, 1], engine_data[:, 0]))
        ]

        zeros_array = np.zeros((engine_data.shape[0], 1))
        # create a new array for thrust_max. here we take the values where throttle=1.0
        thrust_max = np.zeros((engine_data.shape[0], 1))

        # for a given mach, altitude, and hybrid throttle setting, the thrust_max is the value where throttle=1.0
        for i in range(engine_data.shape[0]):
            # find the index of the first instance where throttle=1.0
            index = np.where(
                (engine_data[:, 0] == engine_data[i, 0])
                & (engine_data[:, 1] == engine_data[i, 1])
                & (engine_data[:, 2] == 1.0)
            )[0][0]
            thrust_max[i] = engine_data[index, 3]

        print(Dynamic.Vehicle.Propulsion.THRUST, '--------------------------------------')

        # add inputs and outputs to interpolator
        engine.add_input(
            Dynamic.Atmosphere.MACH,
            engine_data[:, 0],
            units='unitless',
            desc='Current flight Mach number',
        )
        engine.add_input(
            Dynamic.Mission.ALTITUDE,
            engine_data[:, 1],
            units='ft',
            desc='Current flight altitude',
        )
        engine.add_input(
            Dynamic.Vehicle.Propulsion.THROTTLE,
            engine_data[:, 2],
            units='unitless',
            desc='Current engine throttle',
        )
        engine.add_output(
            Dynamic.Vehicle.Propulsion.THRUST,
            engine_data[:, 3],
            units='lbf',
            desc='Current net thrust produced',
        )
        engine.add_output(
            Dynamic.Vehicle.Propulsion.THRUST_MAX,
            thrust_max,
            units='lbf',
            desc='Max net thrust produced',
        )
        engine.add_output(
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE,
            -engine_data[:, 4],
            units='lbm/s',
            desc='Current fuel flow rate ',
        )
        engine.add_output(
            Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN,
            zeros_array,
            units='kW',
            desc='Current electric energy rate',
        )
        engine.add_output(
            Dynamic.Vehicle.Propulsion.NOX_RATE,
            zeros_array,
            units='lb/h',
            desc='Current NOx emission rate',
        )
        engine.add_output(
            Dynamic.Vehicle.Propulsion.TEMPERATURE_T4,
            zeros_array,
            units='degR',
            desc='Current turbine exit temperature',
        )
        return engine

    def get_pre_mission_bus_variables(self):
        """Transfer training data from pre-mission to mission"""
        return vars_to_connect

    def get_controls(self, phase_name):
        """
        Builds dictionary of controls for engine off-design.

        Returns
        -------
        controls : dict
            Dictionary with keys that are names of variables to be controlled and the
            values are dictionaries with the keys `units`, `upper`, and `lower` which states the units of the
            variable to be controlled.
        """
        return {}

    def get_design_vars(self):
        """
        Builds dictionary of design variables for Engine off-design.

        Returns
        -------
        design_vars : dict
            Dictionary with keys that are names of variables to be made design variables
            and the values are dictionaries with the keys `units`, `upper`, `lower`, and `ref`.
        """
        mass_flow_dict = {
            'units': 'lbm/s',
            'upper': 450,
            'lower': 100,
            'ref': 450,
        }  # upper and lower are just notional for now
        design_vars = {
            Aircraft.Engine.DESIGN_MASS_FLOW: mass_flow_dict,
        }

        return design_vars


"""
inputs: mach, altitude
# rename variables
# deal with multiple engines

"""
