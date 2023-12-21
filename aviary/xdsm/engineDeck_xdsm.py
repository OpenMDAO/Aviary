from pyxdsm.XDSM import FUNC, GROUP, METAMODEL, XDSM
from aviary.variable_info.variables import Aircraft, Dynamic

x = XDSM()

use_hybrid_throttle = False
# use_thrust = False

x.add_system('engine', METAMODEL, ['Engine~Interpolator'], stack=True)
x.add_system('scaling', FUNC, [r'Engine~Scaling'], stack=True)

# create inputs
engine_inputs = [
    Dynamic.Mission.ALTITUDE,
    Dynamic.Mission.MACH,
    Dynamic.Mission.THROTTLE,
]
if use_hybrid_throttle:
    engine_inputs.append(r"\textcolor{gray}{"+Dynamic.Mission.HYBRID_THROTTLE+"}")
x.add_input('engine', engine_inputs)

x.add_input("scaling", [
    Dynamic.Mission.MACH,
    Aircraft.Engine.SCALE_FACTOR,
])

# make connections
x.connect("engine", "scaling", [
    'thrust_net_unscaled',
    'thrust_net_max_unscaled',
    'fuel_flow_rate_unscaled',
    'electric_power_unscaled',
    'nox_rate_unscaled',
], stack=True)

# create outputs
x.add_output("scaling", [
    Dynamic.Mission.THRUST,
    Dynamic.Mission.THRUST_MAX,
    Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE,
    Dynamic.Mission.ELECTRIC_POWER,
    Dynamic.Mission.NOX_RATE
], stack=True, side='right')

x.write("engineDeck_xdsm")
# x.write_sys_specs("engineDeck_specs")
