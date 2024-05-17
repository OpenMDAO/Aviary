from aviary.variable_info.variables import Aircraft as av_Aircraft
from aviary.variable_info.variables import Dynamic as av_Dynamic

# ---------------------------
# Aircraft data hierarchy
# ---------------------------


class Aircraft(av_Aircraft):

    class Motor:
        MASS = "aircraft:motor:mass"
        RPM = "aircraft:motor:rpm"
        TORQUE_MAX = "aircraft:motor:torque_max"

    class Gearbox:
        GEAR_RATIO = "aircraft:gearbox:gear_ratio"
        MASS = "aircraft:gearbox:mass"
        TORQUE_MAX = "aircraft:gearbox:torque_max"

    class Prop:
        RPM = "aircraft:prop:rpm"

# ---------------------------
# Mission data hierarchy
# ---------------------------


class Dynamic(av_Dynamic):

    class Mission(av_Dynamic.Mission):
        TORQUE = "dynamic:mission:torque"

        class Motor:
            EFFICIENCY = "dynamic:mission:motor:efficiency"

        class Gearbox():
            EFFICIENCY = "dynamic:mission:gearbox:efficiency"
