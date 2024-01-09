from enum import Enum, auto, unique


class AlphaModes(Enum):
    '''
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
    '''
    DEFAULT = auto()
    ROTATION = auto()
    LOAD_FACTOR = auto()
    FUSELAGE_PITCH = auto()
    DECELERATION = auto()
    REQUIRED_LIFT = auto()
    ALTITUDE_RATE = auto()
    CONSTANT_ALTITUDE = auto()


class AnalysisScheme(Enum):
    """
    AnalysisScheme is used to select from Collocation and shooting.

    COLLOCATION uses the collocation method to optimize all points simultaneously
    and can be run in parallel. However, it requires reasonable initial guesses
    for the trajectory and is fairly sensitive to those initial guesses.

    SHOOTING is a forward in time integration method that simulates the trajectory.
    This does not require initial guesses and will always produce physically valid
    trajectories, even during optimizer failures. The shooting method cannot be run
    in parallel.
    """
    COLLOCATION = auto()
    SHOOTING = auto()


class EquationsOfMotion(Enum):
    """
    Available equations of motion for use during mission analysis
    """
    HEIGHT_ENERGY = 'height_energy'
    TWO_DEGREES_OF_FREEDOM = '2DOF'
    # TODO these are a little out of place atm
    SIMPLE = 'simple'
    SOLVED = 'solved'


@unique
class GASPEngineType(Enum):
    """
    Defines the type of engine to use in GASP-based mass calculations.
    Note that only the value for the first engine model will be used.
    Currenly only the TURBOJET option is implemented, but other types of engines will be added in the future.
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


@unique
class FlapType(Enum):
    """
    Defines the type of flap used on the wing. Used in GASP-based aerodynamics and mass calculations.
    """
    PLAIN = 1
    SPLIT = 2
    SINGLE_SLOTTED = 3
    DOUBLE_SLOTTED = 4
    TRIPLE_SLOTTED = 5
    FOWLER = 6
    DOUBLE_SLOTTED_FOWLER = 7


class LegacyCode(Enum):
    """
    Flag for legacy codebases
    """
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
    """
    SIZING = 'sizing'
    ALTERNATE = 'alternate'
    FALLOUT = 'fallout'


class SpeedType(Enum):
    '''
    SpeedType is used to specify the type of speed being used.
    EAS is equivalent airspeed.
    TAS is true airspeed.
    MACH is mach
    '''
    EAS = auto()
    TAS = auto()
    MACH = auto()


class Verbosity(Enum):
    """
    Sets how much information Aviary outputs when run

    Verbosity levels are based on ubuntu's standard:
    https://discourse.ubuntu.com/t/cli-verbosity-levels/26973
    """
    QUIET = 0
    BRIEF = 1
    VERBOSE = 2
    DEBUG = 3
