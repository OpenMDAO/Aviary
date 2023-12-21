from pyxdsm.XDSM import FUNC, GROUP, XDSM
from aviary.variable_info.variables import Aircraft

x = XDSM()

# Create subsystem components
x.add_system("fus_parameters", FUNC, ["FuselageParameters"])
x.add_system("fus_size", FUNC, ["FuselageSize"])

# create inputs
x.add_input("fus_parameters", [
    # Aircraft.Fuselage.NUM_SEATS_ABREAST,  # option
    # Aircraft.Fuselage.SEAT_WIDTH,  # option
    # Aircraft.Fuselage.NUM_AISLES,  # option
    # Aircraft.Fuselage.AISLE_WIDTH,  # option
    # Aircraft.Fuselage.SEAT_PITCH,  # option
    Aircraft.Fuselage.DELTA_DIAMETER,
    Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH,
])
x.add_input("fus_size", [
    Aircraft.Fuselage.NOSE_FINENESS,
    Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH,
    Aircraft.Fuselage.TAIL_FINENESS,
    Aircraft.Fuselage.WETTED_AREA_FACTOR,
])

# make connections
x.connect("fus_parameters", "fus_size", [
    "cabin_height",
    "cabin_len",
    "nose_height",
])

# create outputs
x.add_output("fus_parameters", [
    Aircraft.Fuselage.AVG_DIAMETER,
    "cabin_height",
    "cabin_len",
    "nose_height",
], side="right")
x.add_output("fus_size", [
    Aircraft.Fuselage.LENGTH,
    Aircraft.Fuselage.WETTED_AREA,
    Aircraft.TailBoom.LENGTH,
], side="right")

x.write("fuselage_size_xdsm")
# x.write_sys_specs("fuselage_size_specs")
