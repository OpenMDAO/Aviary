from pyxdsm.XDSM import FUNC, GROUP, XDSM
from aviary.variable_info.variables import Aircraft, Dynamic, Mission

x = XDSM()

with_EXE_Comps = False

# Create subsystem components
x.add_system("atmos", FUNC, ["USatm"])
x.add_system("fc", FUNC, ["FlightConditions"])
x.add_system("ialpha", FUNC, ["InitAlpha"])
x.add_system("prop", GROUP, ["Propulsion"])
x.add_system("aero", GROUP, ["LowspeedAero"])
x.add_system("eom", GROUP, [r"\textbf{EOM}"])
if with_EXE_Comps:
    x.add_system("exec", FUNC, ["Exec"])
    x.add_system("exec2", FUNC, ["Exec2"])
    x.add_system("exec3", FUNC, ["Exec3"])

# create inputs
x.add_input("atmos", [Dynamic.Mission.ALTITUDE])
x.add_input("fc", ["TAS"])
x.add_input("ialpha", ["i_wing"])
x.add_input("eom", [
    "mass",
    "TAS",
    Dynamic.Mission.FLIGHT_PATH_ANGLE,
    Aircraft.Wing.INCIDENCE,
])
x.add_input("aero", [
    "t_curr",
    "dt_flaps",
    "dt_gear",
    "t_init_flaps",
    "t_init_gear",
    "aircraft:*",
    # Aircraft.Wing.FLAP_DEFLECTION_TAKEOFF,  # flap_defl
    # Aircraft.Wing.HEIGHT,
    # Aircraft.Wing.SPAN,
    # Aircraft.Wing.AREA,
    Dynamic.Mission.ALTITUDE,
    Mission.Takeoff.AIRPORT_ALTITUDE,
    Mission.Design.GROSS_MASS,
    # 'aero_ramps.flap_factor:initial_val',
    # 'aero_ramps.gear_factor:initial_val',
    # 'aero_ramps.flap_factor:final_val',
    # 'aero_ramps.gear_factor:final_val',
])
x.add_input("prop", [
    Dynamic.Mission.ALTITUDE,
    Dynamic.Mission.THROTTLE,
    Aircraft.Engine.SCALE_FACTOR,
])


# make connections
x.connect("atmos", "fc", ["rho", Dynamic.Mission.SPEED_OF_SOUND])
x.connect("fc", "prop", [Dynamic.Mission.MACH])
x.connect("fc", "aero", [Dynamic.Mission.DYNAMIC_PRESSURE, Dynamic.Mission.MACH])
x.connect("prop", "eom", [Dynamic.Mission.THRUST_TOTAL])
x.connect("aero", "eom", [Dynamic.Mission.LIFT, Dynamic.Mission.DRAG])
x.connect("ialpha", "aero", ["alpha"])
x.connect("ialpha", "eom", ["alpha"])
if with_EXE_Comps:
    x.add_input("exec", ["TAS"])
    x.connect("eom", "exec", ["TAS_rate"])
    x.connect("eom", "exec2", ["TAS_rate"])
    x.connect("prop", "exec2", ["fuel_flow_rate_negative_total"])  # mass_rate
    x.connect("exec2", "exec3", ["dt_dv"])

# create outputs
eom_output_list = [
    Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE,
    Dynamic.Mission.ALTITUDE_RATE,
    "distance_rate",
    "alpha_rate",
    "normal_force",
    "fuselage_pitch",
]
if not with_EXE_Comps:
    eom_output_list.append("TAS_rate")
x.add_output("eom", eom_output_list, side="right")

if with_EXE_Comps:
    x.add_output("exec", ["over_a",], side="right")
    x.add_output("exec3", ["dmss_dv",], side="right")
x.add_output("fc", ["EAS",], side="right")

x.write("groundroll_xdsm")
x.write_sys_specs("groundroll_specs")
