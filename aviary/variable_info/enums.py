from enum import Enum, IntEnum, auto, unique


class AircraftTypes(Enum):
    """Aircraft types."""

    TRANSPORT = 'transport'
    BLENDED_WING_BODY = 'BWB'
    # GENERAL_AVIATION = 'GA'  # incomplete in FLOPS, unavailable in GASP


class AlphaModes(Enum):
    """
    AlphaModes is used to specify how angle of attack is defined during
        climb and descent.
    DEFAUT:
        Alpha is an input
    ROTATION
        Alpha is calculated as the initial angle plus the rotation rate
        times the duration of the rotation.
    LOAD_FACTOR
        Alpha is limited to ensure the load factor never exceeds a
        specified maximum.
    FUSELAGE_PITCH
        Alpha is calculated to set a particular floor angle given the
        current flight path angle.
    DECELERATION
        Alpha is calculated to target a specified TAS rate, the default
        is a TAS rate of 0 (Constant TAS).
    REQUIRED_LIFT
        Alpha is calculated such that the aircraft produces a particular
        lifting force.
    ALTITUDE_RATE
        Alpha is calculated to target a specified altitude rate, the default
        is 0 (Constant Altitude).
    """

    DEFAULT = auto()
    ROTATION = auto()
    LOAD_FACTOR = auto()
    FUSELAGE_PITCH = auto()
    DECELERATION = auto()
    REQUIRED_LIFT = auto()
    ALTITUDE_RATE = auto()
    CONSTANT_ALTITUDE = auto()
    FLIGHT_PATH_ANGLE = auto()


class EquationsOfMotion(Enum):
    """Available equations of motion for use during mission analysis."""

    HEIGHT_ENERGY = 'height_energy'
    TWO_DEGREES_OF_FREEDOM = '2DOF'
    SOLVED_2DOF = 'solved_2DOF'
    CUSTOM = 'custom'


@unique
class GASPEngineType(Enum):
    """
    Defines the type of engine to use in GASP-based mass calculations.
    Note that only the value for the first engine model will be used.
    Currently only the TURBOJET and TURBOPROP options are implemented, but other types of engines will be added in the future.
    """

    # Reciprocating engine with carburator
    RECIP_CARB = 1

    # Reciprocating engine with fuel injection
    RECIP_FUEL_INJECT = 2

    # Reciprocating engine with fuel injection and geared
    RECIP_FUEL_INJECT_GEARED = 3

    # Rotary-combustion engine
    ROTARY = 4

    # Turboshaft engine
    TURBOSHAFT = 5

    # Turboprop engine
    TURBOPROP = 6

    # Turbojet or turbofan engine
    TURBOJET = 7

    # Reciprocating engine with carburator; use HOPWSZ (horizontally-opposed piston
    # weight and size) methodology for geometry and mass
    RECIP_CARB_HOPWSZ = 11

    # Reciprocating engine with fuel injection; use HOPWSZ (horizontally-opposed piston
    # weight and size) methodology for geometry and mass
    RECIP_FUEL_INJECT_HOPWSZ = 12

    # Reciprocating engine with fuel injection and geared; use HOPWSZ (horizontally-
    # opposed piston weight and size) methodology for geometry and mass
    RECIP_FUEL_INJECT_GEARED_HOPWSZ = 13

    # Rotary-combustion engine; use RCWSZ (rotary combustion weight and size) methodology
    # for geometry and mass
    ROTARY_RCWSZ = 14

    @classmethod
    def get_element_by_name(cls, val: str):
        return next((c for c in cls if c.name == val), None)


@unique
class FlapType(Enum):
    """Defines the type of flap used on the wing. Used in GASP-based aerodynamics and mass calculations."""

    PLAIN = 1
    SPLIT = 2
    SINGLE_SLOTTED = 3
    DOUBLE_SLOTTED = 4
    TRIPLE_SLOTTED = 5
    FOWLER = 6
    DOUBLE_SLOTTED_FOWLER = 7

    @classmethod
    def get_element_by_name(cls, val: str):
        return next((c for c in cls if c.name == val), None)


class LegacyCode(Enum):
    """Flag for legacy codebases."""

    FLOPS = 'FLOPS'
    GASP = 'GASP'

    def __str__(self):
        return self.value


class ProblemType(Enum):
    """
    ProblemType is used to switch between different combinations of
    design variables and constraints.

    SIZING: Varies the design gross weight and actual gross weight to
    close to design range. This causes the empty weight and the fuel
    weight to change.

    ALTERNATE: Requires a pre-sized aircraft. It holds the design gross
    weight and empty weight constant. It then varies the fuel weight
    and actual gross weight until the range closes to the off-design
    range.

    FALLOUT: Requires a pre-sized aircraft. It holds the design gross
    weight and empty weight constant. Using the specified actual
    gross weight, it will then find the maximum distance the off-design
    aircraft can fly.

    MULTI_MISSION: Similar to a SIZING mission, however it varies the
    design gross weight and actual gross weight across multiple missions
    to and closes design range for each mission. This causes the empty
    weight and the fuel weight to change. The final result will be a
    single empty weight, for all the different missions, and multiple
    values for fuel weight, unique to each mission.
    """

    SIZING = 'sizing'
    ALTERNATE = 'alternate'
    FALLOUT = 'fallout'
    MULTI_MISSION = 'multimission'


class SpeedType(Enum):
    """
    SpeedType is used to specify the type of speed being used.
    EAS is equivalent airspeed.
    TAS is true airspeed.
    MACH is mach.
    """

    EAS = 'EAS'
    TAS = 'TAS'
    MACH = 'mach'

    def __str__(self):
        return self.value


class ThrottleAllocation(Enum):
    """
    Specifies how to handle the throttles for multiple engines.

    FIXED is a user-specified value.
    STATIC is specified by the optimizer as one value for the whole phase.
    DYNAMIC is specified by the optimizer at each point in the phase.
    """

    FIXED = 'fixed'
    STATIC = 'static'
    DYNAMIC = 'dynamic'


class Verbosity(IntEnum):
    """
    Sets how much information Aviary outputs when run.

    Verbosity levels are based on ubuntu's standard:
    https://discourse.ubuntu.com/t/cli-verbosity-levels/26973
    """

    QUIET = 0
    BRIEF = 1
    VERBOSE = 2
    DEBUG = 3

    def __str__(self):
        return str(self.value)

    @classmethod
    def values(cls):
        return {c.value for c in cls}
