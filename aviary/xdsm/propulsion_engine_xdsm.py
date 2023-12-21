from pyxdsm.XDSM import XDSM, GROUP, FUNC, METAMODEL
from aviary.variable_info.variables import Aircraft, Mission, Dynamic

x = XDSM()

use_hybrid_throttle = False

x.add_system("engine", METAMODEL, ["Engine~Interpolator"], stack=True)
x.add_system("scaling", FUNC, ["Engine~Scaling"], stack=True)
x.add_system("mux", FUNC, ["Performance~Mux"])
x.add_system("sum", FUNC, ["Propulsion~Sum"])

# create inputs
prop_eng_inputs = [
    Dynamic.Mission.ALTITUDE,
    Dynamic.Mission.MACH,
    Dynamic.Mission.THROTTLE,
]
if use_hybrid_throttle:
    prop_eng_inputs.append(r"\textcolor{gray}{"+Dynamic.Mission.HYBRID_THROTTLE+"}")
x.add_input("engine",  prop_eng_inputs)

x.add_input("scaling", [
    Dynamic.Mission.MACH,
    Aircraft.Engine.SCALE_FACTOR,
]
)

# make connections
x.connect("engine", "scaling", [
    'thrust_net_unscaled',
    'thrust_net_max_unscaled',
    'fuel_flow_rate_unscaled',
    'electric_power_unscaled',
    'nox_rate_unscaled',
], stack=True)

x.connect("scaling", "mux", [
    Dynamic.Mission.THRUST,
    Dynamic.Mission.THRUST_MAX,
    Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE,
    Dynamic.Mission.ELECTRIC_POWER,
    Dynamic.Mission.NOX_RATE,
], stack=True)

x.connect("mux", "sum", [
    Dynamic.Mission.THRUST,
    Dynamic.Mission.THRUST_MAX,
    Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE,
    Dynamic.Mission.ELECTRIC_POWER,
    Dynamic.Mission.NOX_RATE,
])

# create outputs
x.add_output("engine", [Dynamic.Mission.TEMPERATURE_ENGINE_T4], side="right")
x.add_output("sum", [
    Dynamic.Mission.THRUST_MAX_TOTAL,
    Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
    Dynamic.Mission.ELECTRIC_POWER_TOTAL,
    Dynamic.Mission.NOX_RATE_TOTAL,
], side="right")

x.write("propulsion_engine_xdsm")
# x.write_sys_specs("propulsion_engine_specs")
