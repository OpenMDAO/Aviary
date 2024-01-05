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
    SIZING = auto()
    ALTERNATE = auto()
    FALLOUT = auto()


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


@unique
class GASP_Engine_Type(Enum):
    """
    Defines the type of engine to use in GASP-based mass calculations.
    Note that only the value for the first engine model will be used.
    Currenly only the TURBOJET option is implemented, but other types of engines will be added in the future.
    """

    RECIP_CARB = 1
    """
    Reciprocating engine with carburator
    """
    RECIP_FUEL_INJECT = 2
    """
    Reciprocating engine with fuel injection
    """
    RECIP_FUEL_INJECT_GEARED = 3
    """
    Reciprocating engine with fuel injection and geared
    """
    ROTARY = 4
    """
    Rotary-combustion engine
    """
    TURBOSHAFT = 5
    """
    Turboshaft engine
    """
    TURBOPROP = 6
    """
    Turboprop engine
    """
    TURBOJET = 7
    """
    Turbojet or turbofan engine
    """
    RECIP_CARB_HOPWSZ = 11
    """
    Reciprocating engine with carburator; use HOPWSZ (horizontally-opposed piston weight and size)
    methodology for geometry and mass
    """
    RECIP_FUEL_INJECT_HOPWSZ = 12
    """
    Reciprocating engine with fuel injection; use HOPWSZ (horizontally-opposed piston weight and size)
    methodology for geometry and mass
    """
    RECIP_FUEL_INJECT_GEARED_HOPWSZ = 13
    """
    Reciprocating engine with fuel injection and geared; use HOPWSZ (horizontally-opposed piston weight and size)
    methodology for geometry and mass
    """
    ROTARY_RCWSZ = 14
    """
    Rotary-combustion engine; use RCWSZ (rotary combustion weight and size)
    methodology for geometry and mass
    """


@unique
class Flap_Type(Enum):
    """
    Defines the type of flap used on the wing. Used in GASP-based aerodynamics and mass calculations.
    """

    PLAIN = 1
    """
    Plain flaps
    """
    SPLIT = 2
    """
    Split flaps
    """
    SINGLE_SLOTTED = 3
    """
    Single-slotted flaps
    """
    DOUBLE_SLOTTED = 4
    """
    Double-slotted flaps
    """
    TRIPLE_SLOTTED = 5
    """
    Triple-slotted flaps
    """
    FOWLER = 6
    """
    Fowler flaps
    """
    DOUBLE_SLOTTED_FOWLER = 7
    """
    Double-slotted Fowler flaps
    """


@unique
class Reserves_Type(Enum):
    """
    Defines the type of reserves used. Currently, Set_by_Time is not implemented.
    """

    Set_None = 0
    """
    Do not set it and let initial_guess['reserves'] decide.
    """
    Set_Direct = 1
    """
    Set required fuel reserves directly in lbm (must be larger than 10).
    """
    Set_Fraction = 2
    """
    Set required fuel reserves as a proportion of mission fuel (must be btween -1 and 0).
    """
    Set_by_Time = 3
    """
    Set required fuel reserves as a proportion of time (must be btween 0 and 1 which correspond to 0 and 45 minutes). 
    """
