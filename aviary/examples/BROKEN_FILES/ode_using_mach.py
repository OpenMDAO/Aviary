import unittest

import openmdao.api as om
import aviary.api as av
import numpy as np

from aviary.mission.flops_based.ode.energy_ODE import EnergyODE
# from aviary.models.aircraft.advanced_single_aisle.advanced_single_aisle_data import (
#     detailed_landing_flare,
#     inputs,
#     landing_subsystem_options,
# )
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.functions import set_aviary_initial_values
from aviary.utils.preprocessors import preprocess_options
from aviary.utils.test_utils.default_subsystems import get_default_mission_subsystems
from aviary.validation_cases.validation_tests import do_validation_test
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Dynamic
from aviary.subsystems.propulsion.rc_electric.rc_builder import RCBuilder
from aviary.variable_info.enums import LegacyCode

# class ElectricODETest(unittest.TestCase):

#     def test_case(self):
inputs = AviaryValues()

inputs.set_val(av.Aircraft.Engine.NUM_ENGINES, 2)
# inputs.set_val(av.Aircraft.Wing.AREA, 3.17750636, units = 'ft**2')
inputs.set_val(av.Settings.MASS_METHOD, LegacyCode.FLOPS)
inputs.set_val(av.Mission.Landing.LIFT_COEFFICIENT_MAX, 1.3)
inputs.set_val(av.Aircraft.Battery.MASS, 0.707, units='kg')
inputs.set_val(av.Aircraft.Battery.RESISTANCE, 0.05, units='ohm')
inputs.set_val(av.Aircraft.Battery.VOLTAGE, 22.2, units='V')
inputs.set_val(av.Aircraft.Engine.Motor.IDLE_CURRENT, 0.91, units='A')
inputs.set_val(av.Aircraft.Engine.Motor.PEAK_CURRENT, 120, units='A')
inputs.set_val(av.Aircraft.Engine.Motor.MASS, 0.288, units='kg')
inputs.set_val(av.Aircraft.Engine.Motor.KV, 420, units='rpm/V')
inputs.set_val(Aircraft.Engine.Motor.RESISTANCE, 0.032, units='ohm')
inputs.set_val(av.Aircraft.Engine.Propeller.DIAMETER, 20, units='inch')
inputs.set_val(av.Aircraft.Engine.Propeller.PITCH, 10, units='inch')
inputs.set_val(av.Dynamic.Mission.ALTITUDE, 200, units='ft')
inputs.set_val(av.Dynamic.Mission.VELOCITY, 20, units='m/s')
inputs.set_val(Dynamic.Atmosphere.DENSITY, 1.225, units='kg/m**3')
inputs.set_val(Dynamic.Atmosphere.MACH, 0.031, units='unitless')

prob = om.Problem()

nn = 2
aviary_options = inputs

engines = [RCBuilder()]

preprocess_options(aviary_options, engine_models=engines)


prop = av.CorePropulsionBuilder(engine_models=engines)

prob.model.add_subsystem(
    'energy_ode',
    EnergyODE(
        num_nodes=nn,
        throttle_enforcement='bounded',
        core_subsystems=[prop],
        aviary_options=aviary_options,
    ),
    promotes_inputs=['*'],
    promotes_outputs=['*'],
)

#Does not converge
# prob.model.set_input_defaults(Dynamic.Vehicle.DRAG, val=np.ones(nn) * 19, units='N')
#Does converge
prob.model.set_input_defaults(Dynamic.Vehicle.DRAG, val=np.ones(nn) * 22, units='N')


setup_model_options(prob, aviary_options)

prob.setup(check=False, force_alloc_complex=True)

set_aviary_initial_values(prob, aviary_options)

try:
    prob.run_model()
except: 
    prob.model.list_vars(print_arrays=True, units=True)